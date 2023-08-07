import json
import os
import sqlite3
import tempfile
from pathlib import Path

_CACHE_DIR = None
_CONN = None


def set_cache_dir(path):
    global _CACHE_DIR
    _CACHE_DIR = path


def open_cache():
    global _CACHE_DIR, _CONN
    if _CACHE_DIR is None:
        _CACHE_DIR = Path.home() / ".cache" / "braintrust"

    if _CONN is None:
        oai_cache_path = Path(_CACHE_DIR) / "oai.sqlite"
        os.makedirs(_CACHE_DIR, exist_ok=True)
        _CONN = sqlite3.connect(oai_cache_path)
        _CONN.execute("CREATE TABLE IF NOT EXISTS cache (params text, response text)")
    return _CONN


def run_cached_request(Completion=None, **kwargs):
    if Completion is None:
        # OpenAI is very slow to import, so we only do it if we need it
        import openai

        Completion = openai.Completion

    param_key = json.dumps(kwargs)
    conn = open_cache()
    cursor = conn.cursor()
    resp = cursor.execute("""SELECT response FROM "cache" WHERE params=?""", [param_key]).fetchone()
    if resp:
        return json.loads(resp[0])

    resp = Completion.create(**kwargs).to_dict()

    cursor.execute("""INSERT INTO "cache" VALUES (?, ?)""", [param_key, json.dumps(resp)])
    conn.commit()

    return resp


async def arun_cached_request(Completion=None, **kwargs):
    if Completion is None:
        # OpenAI is very slow to import, so we only do it if we need it
        import openai

        Completion = openai.Completion

    param_key = json.dumps(kwargs)
    conn = open_cache()
    cursor = conn.cursor()
    resp = cursor.execute("""SELECT response FROM "cache" WHERE params=?""", [param_key]).fetchone()
    if resp:
        return json.loads(resp[0])

    resp = (await Completion.acreate(**kwargs)).to_dict()

    cursor.execute("""INSERT INTO "cache" VALUES (?, ?)""", [param_key, json.dumps(resp)])
    conn.commit()

    return resp
