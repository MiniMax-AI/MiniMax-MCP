# Development Standards

## Language and Runtime

- Python >= 3.10
- Package manager: `uv` (preferred) or `pip`
- Dependencies declared in `pyproject.toml`

## Code Quality

- Linter/formatter: `ruff`
- Run before commit: `ruff check minimax_mcp/` and `ruff format minimax_mcp/`
- No shadowing of Python builtins (`format`, `type`, `id`, etc.)
- All `requests.get()` / HTTP calls must include explicit `timeout`
- File I/O must use context managers (`with open(...)`)

## Testing

- Framework: `pytest` with `pytest-cov`
- Test directory: `tests/`
- Run: `pytest tests/ -v`
- Coverage report: `pytest --cov=minimax_mcp --cov-report=term-missing`

## Git

- Conventional commits: `<type>(<scope>): <subject>`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Every commit references an issue when applicable

## CI/CD

- GitHub Actions on push/PR to `main`
- Pipeline: ruff check → ruff format → pytest
- Matrix: Python 3.10, 3.12

## API Client

- All API requests go through `MinimaxAPIClient` (`client.py`)
- Default timeout: `REQUEST_TIMEOUT` (30s) for API calls
- Default timeout: `DOWNLOAD_TIMEOUT` (120s) for file downloads
- Input parameters validated before API calls to avoid wasting credits

## Constants

- All defaults and valid ranges defined in `const.py`
- No magic numbers in tool functions
