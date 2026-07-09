# Hound - Agent Development Guide

This file contains build commands, code style guidelines, and development conventions for agents working on this codebase.

## Build/Lint/Test Commands

### Core Development Commands
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .[dev]  # Install with dev dependencies

# Code formatting and linting
black .                    # Format code (line-length: 100)
ruff check .               # Lint code (line-length: 120, see ruff.toml)
ruff check . --fix         # Auto-fix linting issues
mypy .                     # Type checking (strict mode)

# Testing
pytest                     # Run all tests
pytest -v                  # Verbose test output
pytest -x                  # Stop on first failure
pytest -m "not slow"       # Skip slow tests
pytest -m "not integration"  # Skip tests requiring external services

# Single test execution
pytest tests/test_unified_client.py::TestUnifiedClient::test_init_default_provider
pytest tests/ -k "test_function_name"  # Run tests matching pattern
pytest tests/test_unified_client.py -k "test_provider_switching"

# Coverage testing
pytest --cov=. --cov-report=html --cov-report=term

# Development server
python chatbot/run.py      # Start telemetry UI (default: http://127.0.0.1:5280)
```

### Hound Application Commands
```bash
# Main entry point (from hound/ directory)
./hound.py --help          # Show all available commands

# Project management
./hound.py project create <name> <path>
./hound.py project ls
./hound.py project info <name>

# Graph building
./hound.py graph build <project> --auto --files "src/A.sol,src/B.sol"
./hound.py graph refine <project> SystemArchitecture --iterations 2

# Agent operations
./hound.py agent audit <project> --mode sweep
./hound.py agent audit <project> --mode intuition --time-limit 300
./hound.py agent investigate "query" <project>

# Finalization and reporting
./hound.py finalize <project>
./hound.py report <project> --output /path/to/report.html
```

## Code Style Guidelines

### Imports & Formatting
- Use `ruff` for import sorting; `black` for formatting (line-length: 100)
- Ruff line-length: 120; excludes `chatbot/static` and build artifacts
- Import order: stdlib → third-party → first-party (`llm`, `analysis`, etc.)
- Combine as imports: `from typing import Any, TypeVar`
- Type hints use `|` syntax (Python 3.10+): `str | None`, `dict[str, Any]`

### Type System
- Use `from __future__ import annotations` for forward references
- Strict mypy enabled; use Pydantic BaseModel for data validation
- Generic types: `T = TypeVar('T', bound=BaseModel)`

### Naming Conventions
- Classes: `PascalCase` (e.g., `UnifiedLLMClient`, `AgentParameters`)
- Functions/variables: `snake_case`; Private: `_leading_underscore`
- Files: `snake_case.py`; Constants: `UPPER_SNAKE_CASE`

### Error Handling & Async
- Try/except around external API calls with structured error messages
- Use `async/await` for I/O; HTTP via `httpx`
- Logging via `debug_logger` when available

## Architecture Patterns

### Provider System
- Base class: `BaseLLMProvider` in `llm/base_provider.py`
- Implementations: `OpenAIProvider`, `GeminiProvider`, `AnthropicProvider`, etc.
- Unified client: `UnifiedLLMClient` handles provider switching
- Configuration-driven provider selection per model profile

### Data Models
- Pydantic models for structured data (`schemas.py`)
- Validation and serialization built-in
- Nested models for complex data structures
- JSON schema generation for LLM interactions

### File Organization
```
hound/
├── llm/           # LLM provider implementations
├── analysis/      # Core analysis logic and agents
├── commands/      # CLI command implementations
├── utils/         # Shared utilities (config, JSON, CLI)
├── ingest/        # Code parsing and bundling
├── visualization/ # Graph visualization
├── tests/         # Test suite
└── chatbot/       # Telemetry UI
```

### Testing Conventions
- Test files: `test_*.py` in `tests/` directory
- Test classes: `TestClassName(unittest.TestCase)`
- Test methods: `test_specific_behavior`
- Fixtures: Use `conftest.py` for shared test setup
- Mocking: Mock external LLM APIs in tests
- Markers: `@pytest.mark.slow`, `@pytest.mark.integration`

### Configuration Management
- YAML configuration via `config.yaml`
- Environment variable fallbacks
- Config loader with priority ordering (see `utils/config_loader.py`)
- Profile-based model configuration
- API key management via environment variables

### Session Management
- Session-based audit tracking with per-session planning
- Persistent storage in `~/.hound/projects/`; resume via session IDs
- Token usage tracking per session

## Development Workflow

### Adding New LLM Providers
1. Create provider class inheriting from `BaseLLMProvider`
2. Implement required abstract methods
3. Add provider to `UnifiedLLMClient` factory
4. Add tests in `tests/test_provider_token_tracking.py`

### CLI Command Development
1. Create command file in `commands/` using Typer
2. Follow existing patterns; add help text
3. Test with `./hound.py command --help`

### Debugging and Troubleshooting
- Enable debug mode: `--debug` saves LLM interactions to `.hound_debug/`
- Use telemetry UI: `python chatbot/run.py` with `--telemetry`
- Check token usage via `TokenTracker`

This codebase prioritizes modular design, type safety, and comprehensive testing.