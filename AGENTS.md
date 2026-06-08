# AGENTS.md

This file provides guidance to coding agents when working with code in this repository. `AGENTS.md` is the source of truth; `CLAUDE.md` is a symlink for compatibility.

## Project Overview

Autoevals is a dual-language library (TypeScript + Python) for evaluating AI model outputs. It provides LLM-as-a-judge evaluations, heuristic scorers (Levenshtein distance), and statistical metrics (BLEU). Developed by Braintrust.

## Commands

### TypeScript (in root directory)

```bash
pnpm install --frozen-lockfile          # Install dependencies
pnpm run build                          # Build JS (outputs to jsdist/)
pnpm run test                           # Run all JS tests with vitest
pnpm run test -- js/llm.test.ts                    # Run single test file
pnpm run test -- -t "test name"                    # Run specific test by name
```

### Python (from root directory)

Python dependency management uses `uv` and the project metadata in `pyproject.toml`/`uv.lock`.

```bash
make develop                         # Set up .venv with dev + scipy extras and install pre-commit
source env.sh                        # Activate the .venv
uv sync --extra dev --extra scipy    # Sync local dev dependencies
uv sync --all-extras                 # Sync all optional dependency groups (CI-style)
uv run --extra dev --extra scipy pytest                              # Run Python tests
uv run --extra dev --extra scipy pytest py/autoevals/test_llm.py      # Run single test file
uv run --extra dev --extra scipy pytest -k "test_name"               # Run tests matching pattern
uv run --all-extras python -m build                                   # Build Python package
uv run --all-extras python -m twine check dist/*                      # Check package metadata
```

### Linting

```bash
uv run --extra dev pre-commit run --all-files    # Run all linters (black, ruff, prettier, codespell)
pre-commit run --all-files                       # Also works after make develop/source env.sh
make fixup                                       # Same as above
```

## Architecture

### Dual Implementation Pattern

The library maintains parallel implementations in TypeScript (`js/`) and Python (`py/autoevals/`). Both share:

- The same evaluation templates (`templates/*.yaml`)
- The same `Score` interface: `{name, score (0-1), metadata}`
- The same scorer names and behavior

### Key Modules (both languages)

- `llm.ts` / `llm.py` - LLM-as-a-judge scorers (Factuality, Battle, ClosedQA, Humor, Security, Sql, Summary, Translation)
- `ragas.ts` / `ragas.py` - RAG evaluation metrics (ContextRelevancy, Faithfulness, AnswerRelevancy, etc.)
- `string.ts` / `string.py` - Text similarity (Levenshtein, EmbeddingSimilarity)
- `json.ts` / `json.py` - JSON validation and diff
- `oai.ts` / `oai.py` - OpenAI client wrapper with caching
- `score.ts` / `score.py` - Core Score type and Scorer base class

### Template System

YAML templates in `templates/` define LLM classifier prompts. Templates use Mustache syntax (`{{variable}}`). The `LLMClassifier` class loads these templates and handles:

- Prompt rendering with chain-of-thought (CoT) suffix
- Tool-based response parsing via `select_choice` function
- Score mapping from choice letters to numeric scores

### Python Scorer Pattern

```python
class Scorer(ABC):
    def eval(self, output, expected=None, **kwargs) -> Score      # Sync
    async def eval_async(self, output, expected=None, **kwargs)   # Async
    def __call__(...)  # Alias for eval()
```

### TypeScript Scorer Pattern

```typescript
type Scorer<Output, Extra> = (
  args: ScorerArgs<Output, Extra>,
) => Score | Promise<Score>;
// All scorers are async functions
```

## CI and Releases

- Publishing is handled by trusted publishing workflows documented in `docs/PUBLISHING.md`.
- JavaScript and Python package versions must stay in sync between `package.json` and `py/autoevals/version.py`; CI enforces this via `.github/workflows/version-sync.yaml` and `.github/scripts/check_version_sync.py`.

## Environment Variables

Tests require:

- `OPENAI_API_KEY` or `BRAINTRUST_API_KEY` - For LLM-based evaluations
- `OPENAI_BASE_URL` (optional) - Custom API endpoint

## Testing Notes

- Python tests use `pytest` with `respx` for HTTP mocking
- TypeScript tests use `vitest` with `msw` for HTTP mocking
- Tests that call real LLM APIs need valid API keys
- Test files are colocated: `test_*.py` (Python), `*.test.ts` (TypeScript)
