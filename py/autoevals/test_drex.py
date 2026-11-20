"""Tests for the Drex decision-model scorer.

Offline tests run against a fake Drex (respx). Live tests call the real API and run only when
NACE_API_KEY is set: `NACE_API_KEY=nace_sk_... pytest py/autoevals/test_drex.py -k live`.
"""

import base64
import email.utils
import json
import os
import struct
import sys
import time
import zlib

import httpx
import pytest
import respx

from autoevals import DrexClassifier, DrexError, DrexQuestion
from autoevals.drex import CLIENT_NAME, build_state, is_text_only

BASE = "https://drex.test"
URL = f"{BASE}/v1/systemone"


def png(rgb):
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * 2
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * 2))
        + chunk(b"IEND", b"")
    )


RED, BLUE = png((255, 0, 0)), png((0, 0, 255))


def body(answers, model="drex-max", input_tokens=120, image_tokens=0):
    return {
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "image_tokens": image_tokens, "output_tokens": 0},
        "evaluation_time_ms": 41.5,
        "request_id": "req_test123",
    }


NOUL = {"type": "noul", "noul": 0.9}
CHOICE = {
    "type": "choice",
    "choice": "good",
    "confidence": 0.8,
    "probabilities": {"good": 0.7, "ok": 0.2, "bad": 0.1},
}
SCORE = {
    "type": "score",
    "score": 3.0,
    "confidence": 0.7,
    "legend": ["awful", "poor", "fine", "great", "perfect"],
    "probabilities": {"2": 0.2, "3": 0.6, "4": 0.2},
}


def classifier(question, **kwargs):
    kwargs.setdefault("api_key", "nace_sk_test")
    kwargs.setdefault("base_url", BASE)
    kwargs.setdefault("max_retries", 2)
    return DrexClassifier(question, **kwargs)


def sent(route, index=-1):
    return json.loads(route.calls[index].request.content)


@pytest.fixture
def drex():
    with respx.mock(assert_all_called=False) as mock:
        yield mock


YES_NO = DrexQuestion.noul("Is the output correct?")
CHOICES = DrexQuestion.choice(
    {"good": "Fully right", "ok": None, "bad": "Wrong"}, scores={"good": 1, "ok": 0.5, "bad": 0}
)
RUBRIC = DrexQuestion.score(["awful", "poor", "fine", "great", "perfect"], instructions="Rate the answer.")


@pytest.mark.parametrize("model", ["drex-max", "drex-v2.0"])
def test_noul_choice_score_on_both_models(drex, model):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}, model=model))
    result = classifier(YES_NO, model=model).eval("Paris")
    assert result.score == pytest.approx(0.9)
    assert sent(route)["model"] == model
    assert result.metadata["probabilities"] == {"yes": 0.9, "no": pytest.approx(0.1)}
    assert result.metadata["request_id"] == "req_test123"
    assert result.metadata["model"] == model

    drex.post(URL).respond(json=body({"DrexClassifier": CHOICE}, model=model))
    result = classifier(CHOICES, model=model).eval("Paris")
    # 0.7 * 1 + 0.2 * 0.5 + 0.1 * 0
    assert result.score == pytest.approx(0.8)
    assert result.metadata["choice"] == "good"
    assert result.metadata["probabilities"] == {"good": 0.7, "ok": 0.2, "bad": 0.1}

    drex.post(URL).respond(json=body({"DrexClassifier": SCORE}, model=model))
    result = classifier(RUBRIC, model=model).eval("Paris")
    assert result.score == pytest.approx(0.75)  # 3 / 4
    assert result.metadata["raw_score"] == 3.0
    assert result.metadata["scale"] == [0, 4]
    assert result.metadata["legend"][3] == "great"
    assert result.metadata["probabilities"]["3"] == 0.6


def test_default_model_is_drex_max(drex):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    classifier(YES_NO).eval("x")
    assert sent(route)["model"] == "drex-max"


def test_unknown_model_ids_pass_through(drex):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}, model="drex-v9"))
    classifier(YES_NO, model="drex-v9").eval("x")
    assert sent(route)["model"] == "drex-v9"


def test_mixed_batch_is_one_request_and_counts_usage_once(drex):
    route = drex.post(URL).respond(json=body({"correct": NOUL, "quality": CHOICE, "depth": SCORE}, input_tokens=300))
    scorer = classifier({"correct": YES_NO, "quality": CHOICES, "depth": RUBRIC})
    scores = scorer.eval_many("Paris", input="Capital of France?")
    assert route.call_count == 1
    assert [s.name for s in scores] == ["correct", "quality", "depth"]
    assert set(sent(route)["questions"]) == {"correct", "quality", "depth"}
    assert scores[0].metadata["usage"]["input_tokens"] == 300
    assert scores[1].metadata["usage"] is None and scores[2].metadata["usage"] is None
    assert scores[1].metadata["usage_counted_on"] == "correct"
    with pytest.raises(ValueError, match="eval_many"):
        scorer.eval("Paris")


def test_criteria_only_noul_sends_empty_instructions(drex):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    classifier(DrexQuestion.noul(true="grounded", false="invented")).eval("x")
    question = sent(route)["questions"]["DrexClassifier"]
    assert question["instructions"] == ""
    assert question["criteria"] == {"true": "grounded", "false": "invented"}


def test_choice_null_description_is_kept(drex):
    route = drex.post(URL).respond(json=body({"DrexClassifier": CHOICE}))
    classifier(CHOICES).eval("x")
    assert sent(route)["questions"]["DrexClassifier"]["criteria"]["ok"] is None


def test_state_is_built_from_the_case_fields(drex):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    classifier(YES_NO).eval("answer", "reference", input="question", context="docs")
    assert sent(route)["state"] == {
        "input": "question",
        "output": "answer",
        "expected": "reference",
        "context": "docs",
    }
    classifier(YES_NO, state_fields=("output", "context")).eval("answer", "reference", context="docs")
    assert sent(route)["state"] == {"output": "answer", "context": "docs"}
    classifier(YES_NO).eval("answer")
    assert sent(route)["state"] == "answer"
    assert build_state({"a": 1}) == {"output": {"a": 1}}


def test_direction_reverse_and_threshold(drex):
    drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    defect = DrexQuestion.noul("Does the output contain a defect?", yes_is="fail")
    result = classifier(defect).eval("x")
    assert result.score == pytest.approx(0.1)
    assert result.metadata["yes_is"] == "fail"
    assert result.metadata["probabilities"]["yes"] == 0.9  # the answer itself is not flipped

    drex.post(URL).respond(json=body({"DrexClassifier": SCORE}))
    result = classifier(DrexQuestion.score(RUBRIC.criteria, reverse=True)).eval("x")
    assert result.score == pytest.approx(0.25)

    drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    result = classifier(YES_NO, threshold=0.95).eval("x")
    assert result.score == 0 and result.metadata["raw_value"] == pytest.approx(0.9)
    assert classifier(YES_NO, threshold=0.5).eval("x").score == 1


def test_attribution_headers(drex):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    classifier(YES_NO).eval("x")
    headers = route.calls[0].request.headers
    assert headers["authorization"] == "Bearer nace_sk_test"
    assert headers["x-nace-client"] == CLIENT_NAME
    assert headers["user-agent"] == CLIENT_NAME


def test_one_image_two_images_and_reversed_order(drex):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    scorer = classifier(YES_NO)
    scorer.eval("Is Image 1 red?", images=[RED])
    assert len(sent(route)["images"]) == 1
    scorer.eval("Is Image 1 red?", images=[RED, BLUE])
    forward = sent(route)["images"]
    scorer.eval("Is Image 1 red?", images=[BLUE, RED])
    reverse = sent(route)["images"]
    assert len(forward) == 2 and forward == list(reversed(reverse))
    assert forward[0].startswith("data:image/png;base64,")
    assert base64.b64decode(forward[0].split(",", 1)[1]) == RED


def test_images_on_text_only_model_are_refused_before_sending(drex):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    for model in ("drex-v2.0", "drex-v1.5", "drex-latest", "nace-ai/drex-v2.0"):
        with pytest.raises(ValueError, match="reads text only"):
            classifier(YES_NO, model=model).eval("x", images=[RED])
    assert route.call_count == 0
    assert is_text_only("drex-v2.0") and not is_text_only("drex-max")


def test_remote_image_urls_are_refused(drex):
    with pytest.raises(ValueError, match="doesn't fetch image URLs"):
        classifier(YES_NO).eval("x", images=["https://example.com/a.png"])


def test_workspace_file_reference_passes_through(drex):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    ref = {"type": "workspace_file", "workspace_id": "ws_1", "file_id": "file_1"}
    classifier(YES_NO).eval("x", images=[ref])
    assert sent(route)["images"] == [ref]


def test_more_than_16_questions_on_max_surfaces_drexs_422(drex):
    drex.post(URL).respond(
        422,
        json={
            "error": {
                "type": "invalid_request_error",
                "message": "invalid request",
                "issues": [{"path": "questions", "message": "at most 16 questions allowed for drex-max"}],
            }
        },
    )
    questions = {f"q{i}": DrexQuestion.noul(f"Question {i}?") for i in range(17)}
    with pytest.raises(DrexError) as caught:
        classifier(questions, model="drex-max").eval_many("x")
    assert caught.value.status == 422
    assert "questions: at most 16 questions allowed for drex-max" in str(caught.value)


def test_error_messages(drex):
    drex.post(URL).respond(401, json={"error": {"type": "authentication_error", "message": "Invalid API key"}})
    with pytest.raises(DrexError, match="401") as caught:
        classifier(YES_NO).eval("x")
    assert caught.value.status == 401

    drex.post(URL).respond(402, json={"error": {"type": "insufficient_credit", "message": "No available funds"}})
    with pytest.raises(DrexError) as caught:
        classifier(YES_NO).eval("x")
    assert f"{BASE}/dashboard/billing" in str(caught.value)


@pytest.mark.parametrize("status", [429, 529])
def test_retries_after_retry_after_ms(drex, status, monkeypatch):
    delays = []
    monkeypatch.setattr("time.sleep", delays.append)
    route = drex.post(URL).mock(
        side_effect=[
            httpx.Response(
                status,
                headers={"retry-after-ms": "5"},
                json={"error": {"type": "rate_limit_error", "message": "slow down"}},
            ),
            httpx.Response(200, json=body({"DrexClassifier": NOUL})),
        ]
    )
    result = classifier(YES_NO).eval("x")
    assert route.call_count == 2 and result.score == pytest.approx(0.9)
    assert delays == [pytest.approx(0.005)]  # the header's delay, not the SDK's own backoff


def test_exhausted_retries_are_an_error_not_a_score(drex):
    route = drex.post(URL).respond(
        529, headers={"retry-after-ms": "1"}, json={"error": {"type": "overloaded_error", "message": "busy"}}
    )
    with pytest.raises(DrexError) as caught:
        classifier(YES_NO, max_retries=1).eval("x")
    assert caught.value.status == 529 and route.call_count == 2


def test_openrouter_payload_and_headers(drex):
    url = "https://openrouter.ai/api/v1/systemone"
    route = drex.post(url).respond(json=body({"DrexClassifier": NOUL}))
    scorer = classifier(
        DrexQuestion.noul(true="red"),
        model="nace-ai/drex-max",
        base_url="https://openrouter.ai/api/v1",
    )
    scorer.eval("Is Image 1 red?", images=[RED])
    payload = sent(route)
    assert "images" not in payload
    parts = payload["state"]
    assert parts[0] == {"type": "text", "text": "Is Image 1 red?"}
    assert parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert payload["questions"]["DrexClassifier"]["instructions"] == ""
    headers = route.calls[0].request.headers
    assert headers["http-referer"] == "https://nace.ai"
    assert headers["x-openrouter-title"] == "Drex for Autoevals"
    assert headers["x-nace-client"] == CLIENT_NAME


def test_openrouter_base_url_from_env(drex, monkeypatch):
    monkeypatch.setenv("NACE_BASE_URL", "https://openrouter.ai/api/v1")
    route = drex.post("https://openrouter.ai/api/v1/systemone").respond(json=body({"DrexClassifier": NOUL}))
    DrexClassifier(DrexQuestion.noul(true="red"), api_key="nace_sk_test").eval("x")
    assert route.calls[0].request.headers["x-openrouter-title"] == "Drex for Autoevals"


OPENROUTER = "https://openrouter.ai/api/v1"
OPENROUTER_URL = f"{OPENROUTER}/systemone"


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("NACE_API_KEY", "DREX_API_KEY", "OPENROUTER_API_KEY", "NACE_BASE_URL", "DREX_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("NACE_VERCEL_PROTECTION_BYPASS", raising=False)
    return monkeypatch


def test_openrouter_key_comes_from_api_key_or_openrouter_api_key_never_nace(drex, clean_env):
    route = drex.post(OPENROUTER_URL).respond(json=body({"DrexClassifier": NOUL}))
    clean_env.setenv("NACE_API_KEY", "nace_sk_must_not_leave")
    with pytest.raises(DrexError, match="OPENROUTER_API_KEY"):
        DrexClassifier(YES_NO, base_url=OPENROUTER).eval("x")
    assert route.call_count == 0
    clean_env.setenv("OPENROUTER_API_KEY", "or_env_key")
    DrexClassifier(YES_NO, base_url=OPENROUTER).eval("x")
    assert route.calls[-1].request.headers["authorization"] == "Bearer or_env_key"
    DrexClassifier(YES_NO, base_url=OPENROUTER, api_key="or_explicit").eval("x")
    assert route.calls[-1].request.headers["authorization"] == "Bearer or_explicit"


def test_nace_host_never_uses_the_openrouter_key(drex, clean_env):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    clean_env.setenv("OPENROUTER_API_KEY", "or_must_not_leave")
    with pytest.raises(DrexError, match="API key"):
        DrexClassifier(YES_NO, base_url=BASE).eval("x")
    assert route.call_count == 0
    clean_env.setenv("NACE_API_KEY", "nace_sk_env")
    DrexClassifier(YES_NO, base_url=BASE).eval("x")
    assert route.calls[0].request.headers["authorization"] == "Bearer nace_sk_env"


def test_vercel_bypass_secret_is_never_forwarded(drex, clean_env):
    clean_env.setenv("NACE_VERCEL_PROTECTION_BYPASS", "bypass-secret")
    nace = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    openrouter = drex.post(OPENROUTER_URL).respond(json=body({"DrexClassifier": NOUL}))
    classifier(YES_NO).eval("x")
    classifier(YES_NO, base_url=OPENROUTER).eval("x")
    assert "x-vercel-protection-bypass" not in nace.calls[0].request.headers
    assert "x-vercel-protection-bypass" not in openrouter.calls[0].request.headers


def test_openrouter_host_match_is_exact(drex):
    from autoevals.drex import _is_openrouter

    assert _is_openrouter("https://openrouter.ai/api/v1")
    assert _is_openrouter("https://api.openrouter.ai/v1")
    assert _is_openrouter("https://OpenRouter.ai")
    assert _is_openrouter("https://openrouter.ai./api/v1")
    for other in (
        "https://notopenrouter.ai/api/v1",
        "https://openrouter.ai.evil.test",
        "https://x.test/openrouter.ai",
    ):
        assert not _is_openrouter(other)
    route = drex.post("https://notopenrouter.ai/v1/systemone").respond(json=body({"DrexClassifier": NOUL}))
    classifier(YES_NO, base_url="https://notopenrouter.ai").eval("x")
    assert "x-openrouter-title" not in route.calls[0].request.headers


def test_openrouter_refuses_workspace_files_and_links_its_own_credits(drex):
    scorer = classifier(YES_NO, base_url="https://openrouter.ai/api/v1", model="nace-ai/drex-max")
    ref = {"type": "workspace_file", "workspace_id": "ws_1", "file_id": "file_1"}
    with pytest.raises(ValueError, match="inline images only"):
        scorer.eval("x", images=[ref])
    drex.post("https://openrouter.ai/api/v1/systemone").respond(
        402, json={"error": {"type": "insufficient_credit", "message": "No credits"}}
    )
    with pytest.raises(DrexError, match="openrouter.ai/settings/credits") as caught:
        scorer.eval("x")
    assert "/dashboard/billing" not in str(caught.value)


def test_inline_base64_images_get_their_mime_type_on_openrouter(drex):
    route = drex.post("https://openrouter.ai/api/v1/systemone").respond(json=body({"DrexClassifier": NOUL}))
    jpeg = base64.b64encode(b"\xff\xd8\xff\xe0" + bytes(20)).decode()
    webp = base64.b64encode(b"RIFF\x00\x00\x00\x00WEBP" + bytes(20)).decode()
    other = base64.b64encode(bytes(30)).decode()
    scorer = classifier(YES_NO, base_url="https://openrouter.ai/api/v1", model="nace-ai/drex-max")
    scorer.eval("x", images=[jpeg, webp, other])
    urls = [part["image_url"]["url"] for part in sent(route)["state"][1:]]
    assert [u.split(";")[0] for u in urls] == ["data:image/jpeg", "data:image/webp", "data:application/octet-stream"]


def test_empty_output_is_scored_not_rejected(drex):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    classifier(YES_NO).eval("")
    assert sent(route)["state"] == ""
    with pytest.raises(ValueError, match="Nothing to evaluate"):
        classifier(YES_NO, state_fields=("input",)).eval("x")


def test_a_response_without_a_request_id_is_still_scored(drex):
    payload = body({"DrexClassifier": NOUL})
    del payload["request_id"]
    drex.post(URL).respond(json=payload)
    assert classifier(YES_NO).eval("x").metadata["request_id"] is None


def test_a_huge_retry_after_raises_instead_of_sleeping(drex, monkeypatch):
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    route = drex.post(URL).respond(
        529, headers={"retry-after-ms": "86400000"}, json={"error": {"type": "overloaded_error", "message": "busy"}}
    )
    with pytest.raises(DrexError) as caught:
        classifier(YES_NO).eval("x")
    assert caught.value.status == 529
    assert route.call_count == 1
    assert slept == []


def test_timeouts_are_an_error_not_a_score(drex):
    drex.post(URL).mock(side_effect=httpx.ReadTimeout("too slow"))
    with pytest.raises(DrexError) as caught:
        classifier(YES_NO, max_retries=0).eval("x")
    assert caught.value.status is None


@pytest.mark.asyncio
async def test_async_eval(drex):
    drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    result = await classifier(YES_NO).eval_async("x")
    assert result.score == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_async_errors_and_retries(drex, monkeypatch):
    delays = []

    async def record(delay):
        delays.append(delay)

    monkeypatch.setattr("asyncio.sleep", record)
    busy = {"error": {"type": "overloaded_error", "message": "busy"}}
    route = drex.post(URL).mock(
        side_effect=[
            httpx.Response(529, headers={"retry-after-ms": "5"}, json=busy),
            httpx.Response(200, json=body({"DrexClassifier": NOUL})),
        ]
    )
    assert (await classifier(YES_NO).eval_async("x")).score == pytest.approx(0.9)
    assert route.call_count == 2 and delays == [pytest.approx(0.005)]

    drex.post(URL).respond(402, json={"error": {"type": "insufficient_credit", "message": "No available funds"}})
    with pytest.raises(DrexError) as caught:
        await classifier(YES_NO).eval_async("x")
    assert caught.value.status == 402 and f"{BASE}/dashboard/billing" in str(caught.value)


def test_async_scorer_survives_several_event_loops():
    """A pooled connection kept from a closed loop fails with "Event loop is closed", so this needs a real socket."""
    import asyncio
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    payload = json.dumps(body({"DrexClassifier": NOUL})).encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.send_header("x-request-id", "req_test123")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        scorer = classifier(YES_NO, base_url=f"http://127.0.0.1:{server.server_port}")
        for _ in range(3):
            assert asyncio.run(scorer.eval_async("x")).score == pytest.approx(0.9)
    finally:
        server.shutdown()
        server.server_close()


def test_missing_answer_is_an_error(drex):
    drex.post(URL).respond(json=body({}))
    with pytest.raises(DrexError, match="no answer"):
        classifier(YES_NO).eval("x")


def test_invalid_questions_fail_at_construction(drex):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    with pytest.raises(ValueError, match="cover exactly"):
        classifier(DrexQuestion.choice({"a": None, "b": None}, scores={"a": 1}))
    with pytest.raises(ValueError, match="at least two levels"):
        classifier(DrexQuestion.score(["only"]))
    with pytest.raises(ValueError, match="instructions or criteria"):
        classifier(DrexQuestion.noul())
    with pytest.raises(ValueError, match="threshold"):
        classifier(YES_NO, threshold=2)
    assert route.call_count == 0


def test_request_body_and_headers_follow_the_drex_contract(drex, clean_env):
    clean_env.setenv("NACE_API_KEY", "nace_sk_env")
    route = drex.post(URL).respond(json=body({"DrexClassifier": CHOICE}))
    DrexClassifier(CHOICES, base_url=BASE).eval("Paris", input="Capital of France?")
    request = route.calls[0].request
    assert request.method == "POST" and request.url.path == "/v1/systemone"
    assert sent(route) == {
        "state": {"input": "Capital of France?", "output": "Paris"},
        "questions": {
            "DrexClassifier": {
                "type": "choice",
                "instructions": "",
                "criteria": {"good": "Fully right", "ok": None, "bad": "Wrong"},
            }
        },
        "model": "drex-max",
    }
    assert request.headers["content-type"] == "application/json"
    assert request.headers["accept"] == "application/json"
    assert request.headers["authorization"] == "Bearer nace_sk_env"
    assert request.headers["x-nace-client"] == request.headers["user-agent"] == CLIENT_NAME


def test_request_id_comes_from_the_header_else_the_body(drex):
    payload = body({"DrexClassifier": NOUL})
    drex.post(URL).respond(json=payload, headers={"x-request-id": "req_header"})
    assert classifier(YES_NO).eval("x").metadata["request_id"] == "req_header"
    drex.post(URL).respond(json=payload)
    assert classifier(YES_NO).eval("x").metadata["request_id"] == "req_test123"


def test_request_id_reaches_errors(drex):
    drex.post(URL).respond(
        401, headers={"x-request-id": "req_401"}, json={"error": {"type": "authentication_error", "message": "no"}}
    )
    with pytest.raises(DrexError) as caught:
        classifier(YES_NO).eval("x")
    assert caught.value.request_id == "req_401" and "req_401" in str(caught.value)


def test_authentication_errors_are_not_retried(drex, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: pytest.fail("must not sleep"))
    route = drex.post(URL).respond(401, json={"error": {"type": "authentication_error", "message": "no"}})
    with pytest.raises(DrexError):
        classifier(YES_NO).eval("x")
    assert route.call_count == 1


def test_server_can_forbid_a_retry(drex, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: pytest.fail("must not sleep"))
    route = drex.post(URL).respond(
        503, json={"error": {"type": "overloaded_error", "message": "stop", "retryable": False}}
    )
    with pytest.raises(DrexError):
        classifier(YES_NO).eval("x")
    assert route.call_count == 1


def test_retry_after_seconds_and_http_dates_are_honoured_and_backoff_is_the_fallback(drex, monkeypatch):
    delays = []
    monkeypatch.setattr("time.sleep", delays.append)
    ok = httpx.Response(200, json=body({"DrexClassifier": NOUL}))
    busy = {"error": {"type": "overloaded_error", "message": "busy"}}
    drex.post(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "2"}, json=busy),
            httpx.Response(
                503,
                headers={"retry-after": email.utils.formatdate(time.time() + 3, usegmt=True)},
                json=busy,
            ),
            ok,
        ]
    )
    classifier(YES_NO, max_retries=2).eval("x")
    assert delays[0] == pytest.approx(2.0)
    assert 1.5 < delays[1] <= 3.0
    delays.clear()
    drex.post(URL).mock(side_effect=[httpx.Response(500, json=busy), httpx.Response(502, json=busy), ok])
    classifier(YES_NO, max_retries=2).eval("x")
    assert 0.375 <= delays[0] <= 0.5 and 0.75 <= delays[1] <= 1.0  # 0.5 s and 1 s, minus up to 25% jitter


def test_retry_after_ms_wins_over_retry_after(drex, monkeypatch):
    delays = []
    monkeypatch.setattr("time.sleep", delays.append)
    drex.post(URL).mock(
        side_effect=[
            httpx.Response(
                429, headers={"retry-after-ms": "7", "retry-after": "30"}, json={"error": {"message": "slow"}}
            ),
            httpx.Response(200, json=body({"DrexClassifier": NOUL})),
        ]
    )
    classifier(YES_NO).eval("x")
    assert delays == [pytest.approx(0.007)]


def test_connection_errors_are_retried_then_reported(drex, monkeypatch):
    delays = []
    monkeypatch.setattr("time.sleep", delays.append)
    route = drex.post(URL).mock(
        side_effect=[httpx.ConnectError("refused"), httpx.Response(200, json=body({"DrexClassifier": NOUL}))]
    )
    assert classifier(YES_NO).eval("x").score == pytest.approx(0.9)
    assert route.call_count == 2 and len(delays) == 1
    drex.post(URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(DrexError, match="Connection error") as caught:
        classifier(YES_NO, max_retries=1).eval("x")
    assert caught.value.status is None


def test_timeouts_are_retried_and_the_default_waits_at_least_60_seconds(drex, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    seen = []

    def slow(request):
        seen.append(request.extensions["timeout"]["read"])
        raise httpx.ReadTimeout("too slow", request=request)

    route = drex.post(URL).mock(side_effect=slow)
    with pytest.raises(DrexError, match="timed out") as caught:
        classifier(YES_NO).eval("x")
    assert caught.value.status is None and route.call_count == 3  # the first try and two retries
    assert seen and all(value >= 60 for value in seen)


def test_invalid_timeout_and_retry_settings_fail_at_construction():
    for bad in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="timeout"):
            classifier(YES_NO, timeout=bad)
    with pytest.raises(ValueError, match="max_retries"):
        classifier(YES_NO, max_retries=-1)


def test_a_request_over_4_mib_is_refused_before_sending(drex):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    with pytest.raises(ValueError, match="4 MiB"):
        classifier(YES_NO).eval("x" * (4 * 1024 * 1024 + 1))
    assert route.call_count == 0
    assert classifier(YES_NO).eval("y" * (3 * 1024 * 1024)).score == pytest.approx(0.9)


def test_422_issues_are_listed_as_path_and_message(drex):
    drex.post(URL).respond(
        422,
        json={
            "error": {
                "type": "invalid_request_error",
                "message": "invalid request",
                "issues": [
                    {"path": "questions.q1.criteria", "message": "must not be empty"},
                    {"path": "images", "message": "at most 8 images"},
                ],
            }
        },
    )
    with pytest.raises(DrexError) as caught:
        classifier(YES_NO).eval("x")
    text = str(caught.value)
    assert "422 invalid request" in text
    assert "\n  questions.q1.criteria: must not be empty" in text and "\n  images: at most 8 images" in text


def test_error_bodies_that_are_not_drex_json_still_give_a_message(drex):
    drex.post(URL).respond(502, text="<html>bad gateway</html>")
    with pytest.raises(DrexError, match="502 <html>bad gateway</html>"):
        classifier(YES_NO, max_retries=0).eval("x")
    drex.post(URL).respond(500)
    with pytest.raises(DrexError, match="500 status code \\(no body\\)"):
        classifier(YES_NO, max_retries=0).eval("x")


def test_malformed_and_missing_answers_are_errors(drex):
    drex.post(URL).respond(json={"model": "drex-max", "usage": {}})
    with pytest.raises(DrexError, match="unexpected response"):
        classifier(YES_NO).eval("x")
    drex.post(URL).respond(json=body({"DrexClassifier": {"type": "noul"}}))
    with pytest.raises(DrexError, match="malformed answer"):
        classifier(YES_NO).eval("x")
    drex.post(URL).respond(json=body({"DrexClassifier": {"type": "noul", "noul": "high"}}))
    with pytest.raises(DrexError, match="malformed answer"):
        classifier(YES_NO).eval("x")
    # An answer of another type, or one this client does not know, is no answer to the question asked.
    drex.post(URL).respond(json=body({"DrexClassifier": CHOICE}))
    with pytest.raises(DrexError, match="no answer"):
        classifier(YES_NO).eval("x")
    drex.post(URL).respond(json=body({"DrexClassifier": {"type": "future", "x": 1}}))
    with pytest.raises(DrexError, match="no answer"):
        classifier(YES_NO).eval("x")


def test_usage_fields_are_optional(drex):
    payload = body({"DrexClassifier": NOUL})
    payload["usage"] = {"input_tokens": 5}
    del payload["evaluation_time_ms"]
    drex.post(URL).respond(json=payload)
    metadata = classifier(YES_NO).eval("x").metadata
    assert metadata["usage"] == {"input_tokens": 5, "image_tokens": None, "output_tokens": None}
    assert metadata["evaluation_time_ms"] is None


def test_images_must_be_inline_and_are_never_dropped(drex):
    route = drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    with pytest.raises(ValueError, match="PNG, JPEG and WebP"):
        classifier(YES_NO).eval("x", images=[b"%PDF-1.7 not an image"])
    with pytest.raises(ValueError, match="neither base64 nor a data URL"):
        classifier(YES_NO).eval("x", images=["/tmp/not-base64.png"])
    with pytest.raises(ValueError, match="images\\[0\\]"):
        classifier(YES_NO).eval("x", images=[object()])
    with pytest.raises(ValueError, match="images\\[1\\]"):
        classifier(YES_NO).eval("x", images=[RED, {"type": "url", "url": "https://x.test/a.png"}])
    assert route.call_count == 0
    # A base64 string, a data URL, bytes and an untyped uploaded-file mapping are all sent, in order.
    b64 = base64.b64encode(RED).decode()
    classifier(YES_NO).eval(
        "x", images=[b64, f"data:image/png;base64,{b64}", BLUE, {"workspace_id": "ws_1", "file_id": "file_1"}]
    )
    images = sent(route)["images"]
    assert images[0] == b64 and images[1] == f"data:image/png;base64,{b64}"
    assert base64.b64decode(images[2].split(",", 1)[1]) == BLUE
    assert images[3] == {"type": "workspace_file", "workspace_id": "ws_1", "file_id": "file_1"}


# -- the standard-library transport, used when httpx is not installed ---------------------------


@pytest.fixture
def local_drex(monkeypatch):
    """A real HTTP server whose scripted replies are `(status, headers, json)`; httpx is made unimportable."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    replies = []
    seen = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            content = self.rfile.read(int(self.headers["content-length"]))
            seen.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(content)})
            status, headers, payload = replies.pop(0) if len(replies) > 1 else replies[0]
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setitem(sys.modules, "httpx", None)
    yield f"http://127.0.0.1:{server.server_port}", replies, seen
    server.shutdown()
    server.server_close()


def test_without_httpx_the_standard_library_transport_works(local_drex, monkeypatch):
    base, replies, seen = local_drex
    delays = []
    monkeypatch.setattr("time.sleep", delays.append)
    busy = {"error": {"type": "rate_limit_error", "message": "slow down"}}
    replies += [
        (429, {"retry-after-ms": "5"}, busy),
        (200, {"x-request-id": "req_urllib"}, body({"DrexClassifier": NOUL})),
    ]
    result = classifier(YES_NO, base_url=base).eval("Paris", images=[RED])
    assert result.score == pytest.approx(0.9) and result.metadata["request_id"] == "req_urllib"
    assert delays == [pytest.approx(0.005)] and len(seen) == 2
    assert seen[0]["path"] == "/v1/systemone" and seen[0]["body"]["model"] == "drex-max"
    assert seen[0]["body"]["images"][0].startswith("data:image/png;base64,")
    headers = {name.lower(): value for name, value in seen[0]["headers"].items()}
    assert headers["authorization"] == "Bearer nace_sk_test" and headers["user-agent"] == CLIENT_NAME
    assert headers["x-nace-client"] == CLIENT_NAME


def test_without_httpx_errors_and_connection_failures_are_mapped(local_drex, monkeypatch):
    base, replies, seen = local_drex
    replies.append(
        (
            402,
            {"x-request-id": "req_402"},
            {"error": {"type": "insufficient_credit", "message": "No available funds"}},
        )
    )
    with pytest.raises(DrexError) as caught:
        classifier(YES_NO, base_url=base).eval("x")
    assert caught.value.status == 402 and f"{base}/dashboard/billing" in str(caught.value)
    assert caught.value.request_id == "req_402" and len(seen) == 1
    replies[:] = [
        (422, {}, {"error": {"message": "invalid request", "issues": [{"path": "questions", "message": "too many"}]}})
    ]
    with pytest.raises(DrexError, match="questions: too many"):
        classifier(YES_NO, base_url=base).eval("x")
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(DrexError, match="Connection error"):
        classifier(YES_NO, base_url="http://127.0.0.1:9", max_retries=1).eval("x")


def test_without_httpx_async_runs_the_same_transport(local_drex):
    import asyncio

    base, replies, seen = local_drex
    replies.append((200, {}, body({"DrexClassifier": NOUL})))
    assert asyncio.run(classifier(YES_NO, base_url=base).eval_async("x")).score == pytest.approx(0.9)
    assert len(seen) == 1


def test_the_module_imports_without_httpx_or_a_sdk():
    import subprocess

    code = (
        "import sys; sys.modules['httpx'] = None; sys.modules['nace_sdk'] = None\n"
        "from autoevals.drex import DrexClassifier, DrexQuestion\n"
        "assert DrexClassifier(DrexQuestion.noul('ok?'), api_key='k').model == 'drex-max'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# -- live checks: need a real key, run with `-k live` ---------------------------------

live = pytest.mark.skipif(not os.environ.get("NACE_API_KEY"), reason="NACE_API_KEY is not set")


@live
@pytest.mark.parametrize("model", ["drex-max", "drex-v2.0"])
def test_live_text(model):
    scorer = DrexClassifier(
        DrexQuestion.noul("Does the output state that Paris is the capital of France?"),
        model=model,
        base_url=os.environ.get("NACE_BASE_URL"),
    )
    assert scorer.eval("The capital of France is Paris.").score > 0.8
    assert scorer.eval("The capital of France is Berlin.").score < 0.2


@live
def test_live_images_on_max():
    scorer = DrexClassifier(
        DrexQuestion.noul("Is Image 1 solid red?"), model="drex-max", base_url=os.environ.get("NACE_BASE_URL")
    )
    assert scorer.eval("Look at the images.", images=[RED, BLUE]).score > 0.8
    assert scorer.eval("Look at the images.", images=[BLUE, RED]).score < 0.2


def test_invalid_noul_criteria_and_image_bytes_raise_value_error(drex):
    with pytest.raises(ValueError, match="instructions or criteria"):
        classifier(DrexQuestion.noul(true=""))
    with pytest.raises(ValueError, match="strings or None"):
        classifier(DrexQuestion("noul", "Is it right?", {"true": 1}))
    classifier(DrexQuestion.noul("Is it right?", true="yes", false=None))  # a null outcome is allowed
    with pytest.raises(ValueError, match="PNG, JPEG and WebP"):
        classifier(YES_NO).eval("x", images=[b"not an image"])


def test_openrouter_default_model_is_namespaced_and_bare_ids_are_refused(drex, clean_env):
    route = drex.post(OPENROUTER_URL).respond(json=body({"DrexClassifier": NOUL}))
    DrexClassifier(YES_NO, base_url=OPENROUTER, api_key="or_key").eval("x")
    assert sent(route)["model"] == "nace-ai/drex-max"
    DrexClassifier(YES_NO, base_url=OPENROUTER, api_key="or_key", model="nace-ai/drex-v2.0").eval("x")
    assert sent(route)["model"] == "nace-ai/drex-v2.0"
    for bare in ("drex-max", "drex-v2.0", "drex-latest"):
        with pytest.raises(ValueError, match="namespaced"):
            DrexClassifier(YES_NO, base_url=OPENROUTER, api_key="or_key", model=bare)
    # Nace direct keeps the bare default.
    drex.post(URL).respond(json=body({"DrexClassifier": NOUL}))
    assert classifier(YES_NO).model == "drex-max"


def test_drex_latest_stays_text_only_on_openrouter(drex, clean_env):
    scorer = DrexClassifier(YES_NO, base_url=OPENROUTER, api_key="or_key", model="nace-ai/drex-latest")
    with pytest.raises(ValueError, match="reads text only"):
        scorer.eval("x", images=[RED])


ON_PREM = "https://drex.customer.example"


def test_on_prem_host_uses_the_nace_key_and_plain_nace_shaping(drex, clean_env):
    clean_env.setenv("NACE_API_KEY", "nace_sk_onprem")
    clean_env.setenv("OPENROUTER_API_KEY", "or_must_not_leave")
    route = drex.post(f"{ON_PREM}/v1/systemone").respond(json=body({"DrexClassifier": NOUL}))
    DrexClassifier(YES_NO, base_url=f"{ON_PREM}/").eval("Is Image 1 red?", images=[RED])
    request = route.calls[0].request
    payload = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer nace_sk_onprem"
    assert payload["model"] == "drex-max"
    assert len(payload["images"]) == 1 and payload["state"] == "Is Image 1 red?"
    assert "x-openrouter-title" not in request.headers and "http-referer" not in request.headers
    assert request.headers["x-nace-client"] == CLIENT_NAME
    # `NACE_BASE_URL` alone reaches the same host, and an explicit api_key wins over the environment.
    clean_env.setenv("NACE_BASE_URL", ON_PREM)
    DrexClassifier(YES_NO, api_key="nace_sk_explicit").eval("x")
    assert route.calls[-1].request.headers["authorization"] == "Bearer nace_sk_explicit"


def test_billing_link_follows_the_configured_host(drex, clean_env):
    insufficient = {"error": {"type": "insufficient_credit", "message": "No available funds"}}
    drex.post(f"{ON_PREM}/v1/systemone").respond(402, json=insufficient)
    with pytest.raises(DrexError) as caught:
        DrexClassifier(YES_NO, base_url=ON_PREM, api_key="nace_sk_onprem").eval("x")
    assert f"{ON_PREM}/dashboard/billing" in str(caught.value)
    # With no base URL the link is Nace's console; OpenRouter never gets a Nace link.
    drex.post("https://console.nace.ai/v1/systemone").respond(402, json=insufficient)
    with pytest.raises(DrexError) as caught:
        DrexClassifier(YES_NO, api_key="nace_sk_test").eval("x")
    assert "https://console.nace.ai/dashboard/billing" in str(caught.value)


# -- live OpenRouter checks: need OPENROUTER_API_KEY, never NACE_API_KEY ---------------

live_openrouter = pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY is not set")


@pytest.fixture
def no_nace_key(monkeypatch):
    for name in ("NACE_API_KEY", "DREX_API_KEY", "NACE_BASE_URL", "DREX_BASE_URL", "NACE_VERCEL_PROTECTION_BYPASS"):
        monkeypatch.delenv(name, raising=False)


@live_openrouter
@pytest.mark.parametrize("model", [None, "nace-ai/drex-v2.0", "nace-ai/drex-max"])
def test_live_openrouter_text(no_nace_key, model):
    scorer = DrexClassifier(
        DrexQuestion.noul("Does the output state that Paris is the capital of France?"),
        base_url="https://openrouter.ai/api/v1",
        model=model,
    )
    assert scorer.eval("The capital of France is Paris.").score > 0.8
    assert scorer.eval("The capital of France is Berlin.").score < 0.2


@live_openrouter
def test_live_openrouter_inline_image_on_max(no_nace_key):
    scorer = DrexClassifier(
        DrexQuestion.noul("Is Image 1 solid red?"),
        base_url="https://openrouter.ai/api/v1",
        model="nace-ai/drex-max",
    )
    assert scorer.eval("Look at the images.", images=[RED, BLUE]).score > 0.8
    assert scorer.eval("Look at the images.", images=[BLUE, RED]).score < 0.2
