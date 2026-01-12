# Contributing to Representation Learning Framework

Thank you for your interest in contributing! This guide will help you get started with contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style and Quality](#code-style-and-quality)
- [Adding New Functionality](#adding-new-functionality)
- [Pull Request Process](#pull-request-process)

## Getting Started

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- GCP authentication (for accessing ESP PyPI index):
  ```bash
  gcloud auth login
  gcloud auth application-default login
  ```

### Clone the Repository

```bash
git clone <repository-url>
cd representation-learning
```

## Development Setup

### Install with uv (Recommended)

1. Install keyring with Google Artifact Registry plugin:
   ```bash
   uv tool install keyring --with keyrings.google-artifactregistry-auth
   ```

2. Install the project with all dev/runtime dependencies:
   ```bash
   uv sync --group project-dev
   ```

This will install:
- Base API dependencies
- Training/evaluation runtime dependencies (`pytorch-lightning`, `mlflow`, `wandb`, `esp-data`, etc.)
- Development tools (`pytest`, `ruff`, `pre-commit`, etc.)
- Optional GPU-related packages (when supported)

The `project-dev` dependency group is used by CI and matches the full development environment.

### Install with pip (Alternative)

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install in editable mode with dev extras:
   ```bash
   pip install -e ".[dev]" \
     --extra-index-url https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/
   ```

**Note**: Editable install (`-e`) means changes in the repo are picked up immediately without reinstalling.

## Running Tests

The project uses `pytest` for testing. Tests are organized into three categories:

### Run All Tests

```bash
uv run pytest
```

### Run Specific Test Categories

```bash
# Unit tests
uv run pytest tests/unittests

# Integration tests (excluding slow tests)
uv run pytest tests/integration -m "not slow"

# Consistency tests
uv run pytest tests/consistency --base_folder representation_learning

# Docstring tests
uv run pytest --doctest-modules representation_learning
```

### Run Tests with Coverage

```bash
uv run pytest --cov=representation_learning
```

### Run Tests on Specific Device

```bash
# Run tests on CPU (default)
uv run pytest tests/unittests --device cpu

# Run tests on CUDA (if available)
uv run pytest tests/unittests --device cuda
```

### Run Specific Test Files

```bash
uv run pytest tests/unittests/test_api_registry.py
```

### Run Tests Matching a Pattern

```bash
uv run pytest -k "test_load_model"
```

## Code Style and Quality

### Code Formatting

The project uses **Ruff** for linting and formatting. Ruff replaces `black`, `isort`, and `flake8`.

**Key Style Guidelines:**
- Line length: 88 characters (Ruff default)
- Indentation: 4 spaces (no tabs)
- Quotes: Double quotes for strings
- Import style: Absolute imports preferred over relative
- Type hints: Always use type annotations (PEP 484)
- Docstrings: Google-style docstrings required for all functions/classes

### Running Linters

```bash
# Check code style
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality. Install them:

```bash
uv run pre-commit install
```

Pre-commit hooks will run automatically on `git commit`. You can also run them manually:

```bash
uv run pre-commit run --all-files
```

## Adding New Functionality

### Adding a New Model

1. **Create the model class** in `representation_learning/models/`:
   - Inherit from `ModelBase`
   - Implement required methods (`forward`, `get_embedding_dim`, etc.)
   - See existing models for reference (e.g., `beats_model.py`, `efficientnet.py`)

2. **Register the model class** (if using plugin architecture):
   ```python
   from representation_learning import register_model_class

   @register_model_class
   class MyNewModel(ModelBase):
       name = "my_new_model"
       # ... implementation
   ```

3. **Create a ModelSpec configuration**:
   - Add YAML config in `representation_learning/api/configs/official_models/` (if official)
   - Or create custom YAML config for your use case

4. **Add tests**:
   - Unit tests in `tests/unittests/test_<model_name>.py`
   - Integration tests if needed in `tests/integration/`

5. **Update documentation**:
   - Add model to `docs/supported_models.md`
   - Update API reference if needed

### Adding a New Probe Type

1. **Create the probe class** in `representation_learning/models/probes/`:
   - Inherit from `BaseProbe`
   - Implement required methods

2. **Register the probe**:
   ```python
   from representation_learning.models.probes.utils import register_probe

   @register_probe("my_probe_type")
   class MyProbe(BaseProbe):
       # ... implementation
   ```

3. **Add tests** in `tests/unittests/test_base_probes.py` or create new test file

### Adding New Metrics or Evaluation Functions

1. **Add metric implementation** in `representation_learning/metrics/` or `representation_learning/evaluation/`

2. **Add tests** in `tests/unittests/test_*.py`

3. **Update documentation** if it's a public API

### General Guidelines

- **Follow existing patterns**: Look at similar code in the codebase
- **Write tests first** (TDD approach is encouraged)
- **Keep functions focused**: Single responsibility principle
- **Add type hints**: All functions must have type annotations
- **Write docstrings**: Google-style docstrings for all public functions/classes
- **Handle errors gracefully**: Use specific exception types with informative messages

## Pull Request Process

### Before Submitting

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests locally**:
   ```bash
   # Run all tests
   uv run pytest

   # Run linters
   uv run ruff check .
   uv run ruff format .
   ```

4. **Ensure all tests pass**:
   - Unit tests
   - Integration tests
   - Consistency tests
   - Docstring tests

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: description of your changes"
   ```

   Use clear, descriptive commit messages:
   - `Add: new feature description`
   - `Fix: bug description`
   - `Update: what was updated`
   - `Refactor: what was refactored`
   - `Docs: documentation update`

### Submitting the Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub:
   - Provide a clear title and description
   - Reference any related issues
   - Describe what changes you made and why
   - Include any breaking changes

3. **CI Checks**:
   - The CI will automatically run tests on your PR
   - Ensure all checks pass before requesting review

4. **Code Review**:
   - Address any feedback from reviewers
   - Make additional commits if needed (they will be added to the PR)

### PR Checklist

Before submitting, ensure:

- [ ] All tests pass locally
- [ ] Code follows style guidelines (Ruff checks pass)
- [ ] Type hints are added to all functions
- [ ] Docstrings are added for all public functions/classes
- [ ] Tests are added for new functionality
- [ ] Documentation is updated if needed
- [ ] No breaking changes (or they are documented)
- [ ] Commit messages are clear and descriptive

## Getting Help

- **Documentation**: Check the [docs/](docs/index.md) directory
- **Issues**: Open an issue on GitHub for bugs or feature requests
- **Questions**: Ask in pull request comments or open a discussion

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow

Thank you for contributing! 🎉
