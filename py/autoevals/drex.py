"""Drex decision-model scorer.

`DrexClassifier` asks a Drex decision model (https://console.nace.ai/docs) one typed question about an
evaluation case and turns the answer's probabilities into an Autoevals `Score`. Drex answers with
probabilities, not a verdict, so the score is the probability mass on the outcomes you call a pass, and
the original probabilities stay in `Score.metadata`.

It calls `POST {base_url}/v1/systemone` over plain HTTP (`httpx` when it is installed, which `openai`
brings, else the standard library) and needs no extra package. Credentials come from ``NACE_API_KEY``
(and ``NACE_BASE_URL``, which may name your own Drex deployment) unless passed explicitly.

Example::

    from autoevals.drex import DrexClassifier, DrexQuestion

    grounded = DrexClassifier(
        DrexQuestion.noul("Is every claim in the output supported by the context?"),
        name="Grounded",
    )
    result = grounded.eval(output="Paris is the capital of France.", context="France's capital is Paris.")
    print(result.score)  # probability the answer is "yes"
    print(result.metadata["probabilities"])
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import email.utils
import json
import math
import os
import random
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, NamedTuple
from urllib.parse import urlsplit

from .score import Score, Scorer
from .version import VERSION

CLIENT_NAME = f"autoevals/{VERSION}"
DEFAULT_BASE_URL = "https://console.nace.ai"
SYSTEM_ONE_PATH = "/v1/systemone"
DEFAULT_MODEL = "drex-max"
# OpenRouter namespaces model ids: `drex-max` there is `nace-ai/drex-max`, and a bare id does not exist.
OPENROUTER_MODEL_PREFIX = "nace-ai/"
# Released versions that read text only; Drex enforces every other limit itself.
TEXT_ONLY_MODELS = frozenset({"drex-v1.5", "drex-v2.0", "drex-latest"})
# Drex holds a decision for up to 55 s, so the client must wait longer.
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2
# A server-requested wait longer than this is not slept: the error is raised instead.
MAX_RETRY_WAIT = 60.0
BACKOFF_INITIAL = 0.5
BACKOFF_MAX = 5.0
BACKOFF_JITTER = 0.25
RETRY_STATUSES = frozenset({408, 429, *range(500, 600)})
# Drex refuses a larger request body; images go in inline, so the client says so before sending.
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024
STATE_FIELDS = ("input", "output", "expected", "context")

_BASE64 = re.compile(r"[A-Za-z0-9+/]+={0,2}")


class DrexError(RuntimeError):
    """A request to Drex failed. This is an error in the evaluation, never a low score."""

    def __init__(self, message: str, *, status: int | None = None, request_id: str | None = None):
        super().__init__(message)
        self.status = status
        self.request_id = request_id


@dataclasses.dataclass
class DrexQuestion:
    """One question for Drex and how to turn its answer into a 0-1 score (higher is better).

    Use the `noul`, `choice` and `score` constructors.
    """

    type: Literal["noul", "choice", "score"]
    instructions: str | None = None
    criteria: Any = None
    yes_is: Literal["pass", "fail"] = "pass"
    """`noul` only: whether a yes answer is good. With `"fail"` the score is the probability of no."""
    choice_scores: Mapping[str, float] | None = None
    """`choice` only: the score of each label, from 0 to 1. The score is the probability-weighted sum."""
    reverse: bool = False
    """`score` only: treat the lowest level as best, so the score is one minus the normalized level."""

    @classmethod
    def noul(
        cls,
        instructions: str | None = None,
        *,
        true: str | None = None,
        false: str | None = None,
        yes_is: Literal["pass", "fail"] = "pass",
    ) -> DrexQuestion:
        """A yes/no question. Describe the outcomes with `true` and `false` instead of, or as well as, instructions."""
        criteria = {k: v for k, v in (("true", true), ("false", false)) if v is not None} or None
        return cls("noul", instructions, criteria, yes_is=yes_is)

    @classmethod
    def choice(
        cls,
        choices: Mapping[str, str | None],
        *,
        scores: Mapping[str, float],
        instructions: str | None = None,
    ) -> DrexQuestion:
        """Pick one label. `choices` maps each label to a description or None; `scores` maps each label to 0-1."""
        return cls("choice", instructions, dict(choices), choice_scores=dict(scores))

    @classmethod
    def score(
        cls,
        levels: Sequence[str],
        *,
        instructions: str | None = None,
        reverse: bool = False,
    ) -> DrexQuestion:
        """Rate on an ordered rubric, lowest level first. The score is the expected level divided by the top level."""
        return cls("score", instructions, list(levels), reverse=reverse)

    def validate(self, name: str) -> None:
        if self.type == "noul":
            if self.yes_is not in ("pass", "fail"):
                raise ValueError(f"{name}: yes_is must be 'pass' or 'fail'")
            outcomes = (self.criteria or {}).values()
            if any(v is not None and not isinstance(v, str) for v in outcomes):
                raise ValueError(f"{name}: noul criteria must be strings or None")
            # Drex refuses a yes/no question with neither instructions nor a described outcome.
            if not self.instructions and not any(outcomes):
                raise ValueError(f"{name}: a noul question needs instructions or criteria")
        elif self.type == "choice":
            if not self.criteria:
                raise ValueError(f"{name}: a choice question needs at least one label")
            scores = self.choice_scores or {}
            if set(scores) != set(self.criteria):
                raise ValueError(f"{name}: choice scores must cover exactly the labels {sorted(self.criteria)}")
            if any(not 0 <= v <= 1 for v in scores.values()):
                raise ValueError(f"{name}: choice scores must be between 0 and 1")
        elif self.type == "score":
            if not self.criteria or len(self.criteria) < 2:
                raise ValueError(f"{name}: a score question needs at least two levels")
        else:
            raise ValueError(f"{name}: unknown question type {self.type!r}")

    def wire(self) -> dict[str, Any]:
        """The request form. `instructions` is always a string: OpenRouter requires it, Drex rejects null."""
        question: dict[str, Any] = {"type": self.type, "instructions": self.instructions or ""}
        if self.criteria is not None:
            question["criteria"] = self.criteria
        return question


def is_text_only(model: str) -> bool:
    """Whether `model` is a version known to read text only. Unknown ids are left to Drex."""
    return model.rsplit("/", 1)[-1].lstrip("~") in TEXT_ONLY_MODELS


def _is_openrouter_host(host: str | None) -> bool:
    host = (host or "").lower().rstrip(".")
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def _is_openrouter(base_url: str | None) -> bool:
    return bool(base_url) and _is_openrouter_host(urlsplit(str(base_url)).hostname)


def _env(name: str, legacy: str) -> str | None:
    """A stripped environment value: blank counts as unset, then the legacy `DREX_*` name is tried."""
    for key in (name, legacy):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def build_state(output: Any, expected: Any = None, fields: Sequence[str] = STATE_FIELDS, **kwargs: Any) -> Any:
    """What Drex reads: `output` alone as text, otherwise an object of the case fields that are set."""
    values = {"input": kwargs.get("input"), "output": output, "expected": expected, "context": kwargs.get("context")}
    chosen = {name: values[name] for name in fields if values.get(name) is not None}
    if list(chosen) == ["output"] and isinstance(chosen["output"], str):
        return chosen["output"]
    return chosen


# -- images ---------------------------------------------------------------------------------------


def _media_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _data_url(image: str) -> str:
    """A base64 string as a data URL, typed from its first bytes (OpenRouter needs the type)."""
    if image.startswith("data:"):
        return image
    head = base64.b64decode(image[:24] + "=" * (-len(image[:24]) % 4), validate=False)
    return f"data:{_media_type(head) or 'application/octet-stream'};base64,{image}"


def _prepare_image(image: Any, index: int = 0) -> Any:
    """One `images` entry as the request sends it: base64, a data URL, or a `workspace_file` reference.

    Drex does not fetch URLs or read paths, and images are never dropped: anything else is refused.
    """
    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            raise ValueError(
                "Drex doesn't fetch image URLs. Pass the image bytes, a base64 string or data URL, "
                "or a workspace_file reference."
            )
        if not image.startswith("data:") and (len(image) % 4 or not _BASE64.fullmatch(image)):
            raise ValueError(f"images[{index}] is neither base64 nor a data URL; pass the image bytes instead.")
        return image
    if isinstance(image, (bytes, bytearray)):
        data = bytes(image)
        media_type = _media_type(data)
        if media_type is None:
            raise ValueError("Drex Max reads PNG, JPEG and WebP images only; these bytes are none of them.")
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"The image is {len(data)} bytes, over Drex's 16 MiB per image.")
        return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"
    if isinstance(image, Mapping):
        if image.get("type") == "workspace_file" or (
            "type" not in image and "file_id" in image and "workspace_id" in image
        ):
            return {"type": "workspace_file", **{k: v for k, v in image.items() if k != "type"}}
    raise ValueError(
        f"images[{index}] must be bytes, a base64 string, a data URL or a workspace_file reference, "
        f"not {type(image).__name__}."
    )


def _openrouter_state(state: Any, images: Sequence[Any]) -> list[dict[str, Any]]:
    """OpenRouter takes images as `image_url` parts of a `state` content array, not as `images`."""
    text = state if isinstance(state, str) else json.dumps(state)
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in images:
        if not isinstance(image, str):
            raise ValueError("OpenRouter accepts inline images only, not uploaded workspace files.")
        parts.append({"type": "image_url", "image_url": {"url": _data_url(image)}})
    return parts


# -- responses ------------------------------------------------------------------------------------


def _malformed(name: str) -> DrexError:
    return DrexError(f"Drex returned a malformed answer for question {name!r}")


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("not a number")
    return float(value)


def _normalize(question: DrexQuestion, answer: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    """Return the 0-1 score (higher is better) and the answer's own figures."""
    if question.type == "noul":
        p_yes = _number(answer["noul"])
        probabilities = {"yes": p_yes, "no": 1 - p_yes}
        score = p_yes if question.yes_is == "pass" else 1 - p_yes
        return score, {
            "probabilities": probabilities,
            "yes_is": question.yes_is,
            # Computed here: a deployment may return no confidence on noul answers.
            "confidence": max(p_yes, 1 - p_yes),
        }
    if question.type == "choice":
        probabilities = {str(label): _number(p) for label, p in answer["probabilities"].items()}
        weights = question.choice_scores or {}
        score = sum(p * weights.get(label, 0.0) for label, p in probabilities.items())
        return score, {
            "choice": str(answer["choice"]),
            "confidence": _number(answer["confidence"]),
            "probabilities": probabilities,
            "choice_scores": dict(weights),
        }
    levels = len(question.criteria)
    probabilities = {str(level): _number(p) for level, p in answer["probabilities"].items()}
    raw = _number(answer["score"])
    normalized = raw / (levels - 1)
    score = 1 - normalized if question.reverse else normalized
    return score, {
        "raw_score": raw,
        "scale": [0, levels - 1],
        "legend": [str(item) for item in answer["legend"]],
        "confidence": _number(answer["confidence"]),
        "probabilities": probabilities,
        "reverse": question.reverse,
    }


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


class _Response(NamedTuple):
    status: int
    headers: dict[str, str]  # lower-cased names
    body: bytes


class _TransportError(Exception):
    def __init__(self, message: str, timeout: bool = False):
        super().__init__(message)
        self.timeout = timeout


def _send_urllib(url: str, headers: Mapping[str, str], content: bytes, timeout: float) -> _Response:
    """The standard-library transport, used when `httpx` is not installed."""
    import socket
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, data=content, headers=dict(headers), method="POST")
    message = f"Request timed out after {timeout:g}s"
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _Response(response.status, {k.lower(): v for k, v in response.headers.items()}, response.read())
    except urllib.error.HTTPError as error:
        try:
            body = error.read()
        finally:
            error.close()
        return _Response(error.code, {k.lower(): v for k, v in error.headers.items()}, body)
    except (socket.timeout, TimeoutError):
        raise _TransportError(message, timeout=True) from None
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            raise _TransportError(message, timeout=True) from None
        raise _TransportError(f"Connection error: {error.reason}") from None
    except OSError as error:
        raise _TransportError(f"Connection error: {error}") from None


def _httpx() -> Any:
    try:
        import httpx
    except ImportError:
        return None
    return httpx


def _from_httpx(response: Any) -> _Response:
    return _Response(response.status_code, {k.lower(): v for k, v in response.headers.items()}, response.content)


def _httpx_failure(httpx: Any, error: Exception, timeout: float) -> _TransportError:
    if isinstance(error, httpx.TimeoutException):
        return _TransportError(f"Request timed out after {timeout:g}s", timeout=True)
    return _TransportError(f"Connection error: {type(error).__name__}: {error}")


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """The server's requested wait in seconds: `retry-after-ms`, else `retry-after` (seconds or an HTTP date)."""
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw.strip() or "0")
        except ValueError:
            if name == "retry-after":
                try:
                    parsed = email.utils.parsedate_to_datetime(raw)
                    return max(0.0, parsed.timestamp() - time.time())
                except (TypeError, ValueError, OverflowError):
                    pass
            continue
        if math.isfinite(value) and value >= 0:
            return value * scale
    return None


def _json_or_text(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return body.decode("utf-8", "replace")


def _error_message(body: Any) -> str | None:
    """The server's message, from `error` (string or object), `message` or `detail`."""
    if isinstance(body, str):
        return body[:200] or None
    if not isinstance(body, dict):
        return None
    error, message, detail = body.get("error"), body.get("message"), body.get("detail")
    if isinstance(error, str):
        return error
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(message, str):
        return message
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict) and isinstance(detail.get("message"), str):
        return detail["message"]
    return None


def _body_request_id(body: Any) -> str | None:
    value = body.get("request_id") if isinstance(body, dict) else None
    return value if isinstance(value, str) else None


class DrexClassifier(Scorer):
    """Score an evaluation case with a Drex decision model.

    Args:
        question: A `DrexQuestion`, or a mapping of names to questions to answer in one request
            (see `eval_many`).
        name: The score's name. Defaults to `DrexClassifier`.
        model: A Drex model id, passed through unchanged. Defaults to `drex-max`, which also reads
            images; use `drex-v2.0` for large text-only rubrics if Drex rejects the question count. On OpenRouter
            the default is `nace-ai/drex-max` and a bare id is refused.
        images: Images every question sees (`drex-max` only), or None. Can also be passed per call as `images=`.
        state_fields: Which of `input`, `output`, `expected` and `context` Drex reads.
        threshold: When set, the score is 1 if the 0-1 score reaches it and 0 otherwise; the
            unthresholded value stays in `metadata["raw_value"]`.
        api_key, base_url: Default to `NACE_API_KEY` and `NACE_BASE_URL` (Nace direct). Any base URL that is
            not OpenRouter is a Drex deployment, Nace's or your own. Point `base_url` at
            `https://openrouter.ai/api/v1` to use OpenRouter (model ids `nace-ai/drex-max`);
            the key then comes from `api_key` or `OPENROUTER_API_KEY` and never from `NACE_API_KEY`.
        timeout: Seconds per attempt. At least 60 is advised.
        max_retries: Retries on 408, 429 and 5xx (529 included) and on connection errors, honouring
            `retry-after-ms`, then `retry-after`. A requested wait over 60 s is not slept: the error is raised.
    """

    def __init__(
        self,
        question: DrexQuestion | Mapping[str, DrexQuestion],
        *,
        name: str | None = None,
        model: str | None = None,
        images: Sequence[Any] | None = None,
        state_fields: Sequence[str] = STATE_FIELDS,
        threshold: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.questions: dict[str, DrexQuestion] = (
            {name or "DrexClassifier": question} if isinstance(question, DrexQuestion) else dict(question)
        )
        if not self.questions:
            raise ValueError("At least one question is required")
        for key, item in self.questions.items():
            item.validate(key)
        unknown = [f for f in state_fields if f not in STATE_FIELDS]
        if unknown:
            raise ValueError(f"Unknown state fields {unknown}; choose from {list(STATE_FIELDS)}")
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive, finite number of seconds")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.name = name or (next(iter(self.questions)) if len(self.questions) == 1 else "DrexClassifier")
        self.base_url = base_url or _env("NACE_BASE_URL", "DREX_BASE_URL")
        self.openrouter = _is_openrouter(self.base_url)
        if model is not None and not model.strip():
            raise ValueError("model must not be empty")
        if self.openrouter and model and "/" not in model:
            raise ValueError(
                f"OpenRouter model ids are namespaced: use {OPENROUTER_MODEL_PREFIX}{model!r} "
                f"(for example {OPENROUTER_MODEL_PREFIX}{DEFAULT_MODEL}), not {model!r}."
            )
        self.model = model or (OPENROUTER_MODEL_PREFIX + DEFAULT_MODEL if self.openrouter else DEFAULT_MODEL)
        self.images = list(images) if images else None
        self.state_fields = tuple(state_fields)
        self.threshold = threshold
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Any = None

    def _name(self) -> str:
        return self.name

    # -- request ----------------------------------------------------------------------------------

    def _url(self) -> str:
        base = (self.base_url or DEFAULT_BASE_URL).rstrip("/")
        if self.openrouter:
            # OpenRouter documents `.../api/v1`; the route is `/v1/systemone` under `.../api`.
            base = base.removesuffix("/v1")
        return base + SYSTEM_ONE_PATH

    def _headers(self) -> dict[str, str]:
        if self.openrouter:
            # The key follows the host: never fall back to NACE_API_KEY for OpenRouter.
            key = self.api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not key:
                raise DrexError("No OpenRouter API key was provided. Pass api_key or set OPENROUTER_API_KEY.")
        else:
            key = (self.api_key or _env("NACE_API_KEY", "DREX_API_KEY") or "").strip()
            if not key:
                raise DrexError("No API key was provided. Pass api_key or set the NACE_API_KEY environment variable.")
        if not key.isascii() or not key.isprintable() or " " in key:
            raise DrexError("The API key must contain only printable ASCII characters without whitespace.")
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Nace-Client": CLIENT_NAME,
            "User-Agent": CLIENT_NAME,
        }
        if self.openrouter:
            headers.update({"HTTP-Referer": "https://nace.ai", "X-OpenRouter-Title": "Drex for Autoevals"})
        return headers

    def _payload(self, output: Any, expected: Any, kwargs: dict[str, Any], names: Sequence[str]) -> dict[str, Any]:
        images = kwargs.get("images", self.images)
        state = build_state(output, expected, self.state_fields, **kwargs)
        if state == {}:
            raise ValueError("Nothing to evaluate: pass at least one of output, expected, input or context")
        prepared = [_prepare_image(image, index) for index, image in enumerate(images or ())]
        if prepared and is_text_only(self.model):
            raise ValueError(
                f"{self.model} reads text only, so it can't take images. Use an image model such as model='drex-max'."
            )
        if prepared and self.openrouter:
            state, prepared = _openrouter_state(state, prepared), []
        payload: dict[str, Any] = {
            "state": state,
            "questions": {n: self.questions[n].wire() for n in names},
            "model": self.model,
        }
        if prepared:
            payload["images"] = prepared
        return payload

    def _encode(self, payload: dict[str, Any]) -> bytes:
        try:
            content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError(f"The case can't be sent as JSON: {error}") from error
        # OpenRouter's own limit is not Drex's, so only Drex deployments are checked here.
        if not self.openrouter and len(content) > MAX_BODY_BYTES:
            raise ValueError(
                f"The request is {len(content) / 1048576:.1f} MiB, over Drex's 4 MiB limit. "
                "Send fewer or smaller images or less text; the scorer does not upload files."
            )
        return content

    # -- retries and errors -----------------------------------------------------------------------

    def _delay(self, attempt: int, response: _Response | None) -> float | None:
        """Seconds to wait before retry number `attempt`, or None to stop and raise."""
        if attempt > self.max_retries:
            return None
        if response is not None:
            if response.status not in RETRY_STATUSES:
                return None
            body = _json_or_text(response.body)
            error = body.get("error") if isinstance(body, dict) else None
            if isinstance(error, dict) and error.get("retryable") is False:
                return None
            wait = _retry_after(response.headers)
            if wait is not None:
                return wait if wait <= MAX_RETRY_WAIT else None
        backoff = min(BACKOFF_INITIAL * 2 ** (attempt - 1), BACKOFF_MAX)
        return backoff * (1 - random.random() * BACKOFF_JITTER)

    def _api_error(self, response: _Response) -> DrexError:
        body = _json_or_text(response.body)
        message = f"{response.status} {_error_message(body) or 'status code (no body)'}"
        error = body.get("error") if isinstance(body, dict) else None
        issues = error.get("issues") if isinstance(error, dict) else None
        if isinstance(issues, list):
            message += "".join(
                f"\n  {issue.get('path', '')}: {issue.get('message', '')}"
                for issue in issues
                if isinstance(issue, dict)
            )
        if response.status == 402:
            if self.openrouter:
                message += "\nAdd credits at https://openrouter.ai/settings/credits."
            else:
                host = (self.base_url or DEFAULT_BASE_URL).rstrip("/")
                message += f"\nAdd funds or a card at {host}/dashboard/billing."
        request_id = response.headers.get("x-request-id") or _body_request_id(body)
        if request_id:
            message += f" (request_id={request_id})"
        return DrexError(message, status=response.status, request_id=request_id)

    def _parse(self, response: _Response) -> dict[str, Any]:
        body = _json_or_text(response.body)
        request_id = response.headers.get("x-request-id") or _body_request_id(body)
        if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
            raise DrexError(
                "Drex returned an unexpected response (no answers).", status=response.status, request_id=request_id
            )
        return {**body, "request_id": request_id}

    def _post_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        url, headers, content = self._url(), self._headers(), self._encode(payload)
        httpx = _httpx()
        attempt = 0
        while True:
            attempt += 1
            try:
                if httpx is None:
                    response = _send_urllib(url, headers, content, self.timeout)
                else:
                    if self._client is None:
                        self._client = httpx.Client()
                    try:
                        response = _from_httpx(
                            self._client.post(url, content=content, headers=headers, timeout=self.timeout)
                        )
                    except httpx.RequestError as error:
                        raise _httpx_failure(httpx, error, self.timeout) from None
            except _TransportError as error:
                delay = self._delay(attempt, None)
                if delay is None:
                    raise DrexError(str(error)) from None
            else:
                if 200 <= response.status < 300:
                    return self._parse(response)
                delay = self._delay(attempt, response)
                if delay is None:
                    raise self._api_error(response)
            time.sleep(delay)

    async def _post_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        url, headers, content = self._url(), self._headers(), self._encode(payload)
        httpx = _httpx()
        # An httpx connection pool is bound to the event loop that first used it, so a client kept across
        # `asyncio.run()` calls fails with "Event loop is closed". Hence one per call.
        client = httpx.AsyncClient() if httpx is not None else None
        attempt = 0
        try:
            while True:
                attempt += 1
                try:
                    if client is None:
                        response = await asyncio.to_thread(_send_urllib, url, headers, content, self.timeout)
                    else:
                        try:
                            response = _from_httpx(
                                await client.post(url, content=content, headers=headers, timeout=self.timeout)
                            )
                        except httpx.RequestError as error:
                            raise _httpx_failure(httpx, error, self.timeout) from None
                except _TransportError as error:
                    delay = self._delay(attempt, None)
                    if delay is None:
                        raise DrexError(str(error)) from None
                else:
                    if 200 <= response.status < 300:
                        return self._parse(response)
                    delay = self._delay(attempt, response)
                    if delay is None:
                        raise self._api_error(response)
                await asyncio.sleep(delay)
        finally:
            if client is not None:
                await client.aclose()

    def _scores(self, response: dict[str, Any], names: Sequence[str]) -> list[Score]:
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        shared = {
            "input_tokens": usage.get("input_tokens"),
            "image_tokens": usage.get("image_tokens"),
            "output_tokens": usage.get("output_tokens"),
        }
        request_id = response["request_id"]
        scores = []
        for index, question_name in enumerate(names):
            question = self.questions[question_name]
            answer = response["answers"].get(question_name)
            if not isinstance(answer, dict) or answer.get("type") != question.type:
                raise DrexError(f"Drex returned no answer for question {question_name!r}", request_id=request_id)
            try:
                value, figures = _normalize(question, answer)
            except (KeyError, TypeError, ValueError, AttributeError):
                raise _malformed(question_name) from None
            value = _clamp(value)
            final = value if self.threshold is None else float(value >= self.threshold)
            metadata = {
                "question": question_name,
                "type": question.type,
                "direction": "higher is better (probability of a pass)",
                **figures,
                "raw_value": value,
                "threshold": self.threshold,
                "model": response.get("model"),
                "requested_model": self.model,
                "request_id": request_id,
                "evaluation_time_ms": response.get("evaluation_time_ms"),
                # One request serves every question; count its tokens once, on the first score.
                "usage": shared if index == 0 else None,
                "usage_counted_on": names[0],
            }
            scores.append(
                Score(name=question_name if len(self.questions) > 1 else self.name, score=final, metadata=metadata)
            )
        return scores

    def _single(self) -> str:
        if len(self.questions) != 1:
            raise ValueError("This scorer has several questions: use eval_many(), which answers them in one request")
        return next(iter(self.questions))

    # -- Scorer API -------------------------------------------------------------------------------

    def _run_eval_sync(self, output: Any, expected: Any = None, **kwargs: Any) -> Score:
        self._single()
        return self.eval_many(output, expected, **kwargs)[0]

    async def _run_eval_async(self, output: Any, expected: Any = None, **kwargs: Any) -> Score:
        self._single()
        return (await self.eval_many_async(output, expected, **kwargs))[0]

    def eval_many(self, output: Any, expected: Any = None, **kwargs: Any) -> list[Score]:
        """Answer every question about one case in a single request; one `Score` per question, in order."""
        names = list(self.questions)
        payload = self._payload(output, expected, kwargs, names)
        return self._scores(self._post_sync(payload), names)

    async def eval_many_async(self, output: Any, expected: Any = None, **kwargs: Any) -> list[Score]:
        names = list(self.questions)
        payload = self._payload(output, expected, kwargs, names)
        return self._scores(await self._post_async(payload), names)


__all__ = ["DrexClassifier", "DrexError", "DrexQuestion", "build_state", "is_text_only"]
