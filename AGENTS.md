# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

DeepSeek FinRobot is a Chinese financial AI agent platform built on the DeepSeek LLM API. It is a **CLI-only Python library** — no web server, database, or Docker required. See `README.md` for full architecture and usage examples.

### Key dependency caveat

The code imports `autogen` (the v0.2 API of pyautogen), but `requirements.txt` specifies `pyautogen>=0.8.5` which installs the newer v0.4+ package that no longer exposes the `autogen` module. You **must** install `pyautogen==0.2.35` to get the correct `import autogen` compatibility. The update script handles this automatically.

### Running tests

```bash
python3 -m pytest tests/ -v
```

All 7 tests pass without a DeepSeek API key. Tests that would call the API are skipped when `DEEPSEEK_API_KEY` is not set.

For coverage: `python3 -m pytest tests/ -v --cov=deepseek_finrobot --cov-report=term`

### Linting

No linter is configured in the repo. You can use `ruff check deepseek_finrobot/ tests/` for basic checks. Pre-existing lint findings are expected.

### Running the CLI

```bash
python3 -m deepseek_finrobot.cli --help
```

Full CLI commands (predict, industry, portfolio, etc.) require a valid `DEEPSEEK_API_KEY` in either the environment or `config_api_keys.json` at project root (copy from `config_api_keys_sample`).

**Important**: The CLI's `predict` command (and other agent commands) uses AutoGen's conversation loop, which may run for many minutes due to auto-reply behavior. For quick validation, use the Python API directly (e.g. `get_completion()` from `deepseek_finrobot.openai_adapter`).

### CLI config_api_keys.json setup

The CLI requires `config_api_keys.json` at the project root. If `DEEPSEEK_API_KEY` is set in the environment, create it with:

```python
import json, os
with open('config_api_keys.json', 'w') as f:
    json.dump({'DEEPSEEK_API_KEY': os.environ['DEEPSEEK_API_KEY']}, f)
```

Remember to delete this file before committing (it's not gitignored).

### External dependencies

- **DeepSeek API**: Requires `DEEPSEEK_API_KEY` for any LLM-powered agent functionality.
- **AKShare**: Fetches live Chinese financial data from public APIs (no key needed, but requires internet access). The primary `akshare` endpoint for industry lists sometimes fails with `RemoteDisconnected`; the code has built-in fallback logic that usually succeeds on retry.
