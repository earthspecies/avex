# A Template for Python-Based Projects at Earth Species Project

For 2025, ESP is focusing on credibility. In terms of code, credibility relies on **trust**, and trust in code comes from structure and correctness.

The main idea behind this repository is to ensure structure, consistency, and correctness for ESP Python projects.

This repository can be used as a starting point for new projects, and it helps maintain good practices through automatic testing and fixing. It can also be used to improve existing projects.

This new coding structure is based on three main principles:

1. Documentation.

2. Consistency.

3. Testing.

All of these are automatically checked and/or updated through a two-step process:

1. **Pre-commit tests (using pre-commit hooks):** These tests and fixes happen every time you commit files. These quick tests ensure documentation and consistency. They should be fixed locally before pushing the code and serve as a guide for producing consistent code that follows good practices.

2. **GitHub Continuous Integration (CI) tests:** These tests ensure documentation, consistency, and testing, and are required before merging any Pull Request. They can also be run manually before pushing, using `pytest`.

# Your Feedback 🗣️

This template aims to facilitate the life of ESP projects. If you end up struggling due to something, or if you think that something should be changed or improved, **please open an issue!** This template is meant to evolve.

# How to Use This Template?

There are two main use cases:

1. Starting a new project from scratch.

2. Adapting an existing project.

## Starting a New Project from Scratch

**Files that require editing contain TODO (OSS) hints that you can search for.**

1. **Create from a template:** On the top right of this repository's webpage, click on *Use this template*. Then, select *Create a new repository* and proceed as you normally would. This will create a new repository as a full copy of this one. You should see Continuous Integration start as an orange dot near the last merge in the branch (in the file explorer table header row).

2. **Customize your project information:** Open and edit the `pyproject.toml` file. You'll find fields under `[project]`, `[dependency-groups]`, `[project.urls]`, and `[tool.hatch.build.targets.wheel]` that need to be edited with the correct values. This includes project names, dependencies, descriptions, and links. Feel free to remove any descriptors that don't apply. The `packages` field under `[tool.hatch.build.targets.wheel]` should point to the directory containing your library (or libraries). If you don't need to build your code as a package, you can remove all the building information (leaving it by default won't hurt!).

3. **Explore the structure:** Open `my_dummy_library` and look through the files to see how your code should be structured. Do the same with the `tests` folder.

4. **Verify GitHub CI:** Check that the GitHub Continuous Integration is working. The orange dot from step one should now be a green checkmark, meaning the tests have passed. If it's a red cross, click on it for error details. If this happens, please create an issue in this repository, as it shouldn't occur.

5. **Set up your Python environment:** Install the package and required dependencies. You can use your preferred tool, but we recommend `uv`. Make sure to also install the `requirements-dev` packages. **`Important: Never manually update the requirements files.  Only update the pyproject.toml file, because the pre-commit hook automatically generates them!`**

6. **Install pre-commit:** `pre-commit` will check that you're following the rules (described below) for the files you stage with `git add` before committing. First, run `pre-commit install`. After that, every commit will trigger the consistency tests.

7. **Run the tests:** To see how the tests work, run both the docstring tests and the integration/unit tests. Run the unit tests with `pytest tests --base_folder my_dummy_library` and the docstring tests with `pytest --doctest-modules my_dummy_library`. The docstring test command might create a `.pytest_cache` folder, which you may need to delete before running `pytest tests` (if you ran the docstring tests first). Everything should pass.

8. **Prepare for your project:** You can now delete the `my_dummy_library` folder and replace it with your `your_library` folder.  Remember to update `.github/workflows/pythonapp.yaml` at the line `pytest --doctest-modules my_dummy_library` with your new library name.  You can also delete the example test files `test_VanillaNN.py` in `tests/integration` and `test_linear.py` in `tests/unittests` and replace them with your own test files.  (You can also keep the example files for reference and add your library.)  You can also delete or replace the `README.md` file.

9. **Add, commit, and push!** GitHub CI might complain if there are no tests, but this will change once you add your test files.

## Adapting an Existing Project

This assumes that you haven't set up GitHub continuous integration workflows before. If you have, you'll need to merge your existing workflows with these ones. This process depends heavily on the project, and you might need help from the engineering team. If these simplified steps aren't clear enough for your project, please contact the engineering team.

1. Clone this repository.

2. Create a new branch in your repository.

3. **Migrate to `pyproject.toml:`** If you don't have a `pyproject.toml` file, copy the one from this repository into yours. If you do have one, make sure that **all** the fields from this `pyproject.toml` are copied into yours.  Make sure the fields under `[project]`, `[dependency-groups]`, `[project.urls]`, and `[tool.hatch.build.targets.wheel]` are filled with values that reflect your project.

4. **Copy files and folders:** Copy `.github/workflows`, `.dict-allowed.txt`, `conftest.py`, `tests`, and `.pre-commit-config.yaml` to the root of your GitHub repository.

5. **Set up the tests:** First, initialize pre-commit (after updating your environment with the new `pyproject.toml` and the development tools): `pre-commit install`.  Then, update `.github/workflows/pythonapp.yaml` with your library's name, replacing  `my_dummy_library` at the line `pytest --doctest-modules my_dummy_library`.  Replace the example test files `test_VanillaNN.py` in `tests/integration` and `test_linear.py` in `tests/unittests` with your own test files.  Your test files should be named `test_*` or `check_*`.

6. **Push the copied files:** Use `git add` and `push` to upload the copied files. This will trigger the GitHub CI, but it will likely fail initially because your code isn't properly formatted yet.

7. **Refactor your code:** This is a long process. We recommend fixing your files one by one, following the rules. To see the specific errors, make a change in one of your library files, use `git add`, and try to commit. Pre-commit will show you a list of changes to make.  Look at the example files in `my_dummy_library` to see the correct formatting. After making the changes, add the file again, commit, and move on to the next one.

8. **Open a Pull Request:** Once you've made the changes, open a Pull Request and check for any CI errors (at the bottom of the PR).  You can test the CI tests locally by running  `pytest tests` and `pytest --doctest-modules my_dummy_library`. If these tests pass locally and pre-commit isn't complaining, then the GitHub CI should also pass.

# Explanation of The Tools

This template uses two tools: [ruff](https://github.com/astral-sh/ruff) for code homogeneity and small bugs catching and [pytest](https://docs.pytest.org/en/stable/) for docstring, unitary and integration testing. This section will give a few details about docstrings, ruff linting and unitary/integration tests.

## Documentation of Code Functionality: Docstrings and Doctest

A **docstring** serves as an essential component of code documentation, providing explanatory text that details the purpose and usage of functions, modules, classes, and methods.  Adhering to a standardized format enhances readability and facilitates the use of automated documentation tools. In principel, we want every class and function to have a docstring. These, of course, can be rather short for simple functions. If you really don't want to write a doctstring for a class or function, you can add `# noqa: DOC101` after the code of this class/function.

One widely adopted convention for structuring docstrings is the **NumPy style**. This format promotes clarity and consistency through the use of specific sections.

Consider the following function with a NumPy-style docstring:

```python
def add_two_numbers(number1 : float, number2 : float):
  """Computes the sum of two numerical inputs.

  Parameters
  ----------
  number1 : int
      The first addend.
  number2 : int
      The second addend.

  Returns
  -------
  int
      The arithmetic sum of the two input numbers.

  Examples
  --------
  >>> add_two_numbers(5, 3)
  8
  >>> add_two_numbers(-1, 10)
  9
  """
  return number1 + number2
  ```

Doctest is an integrated Python module that enables the execution of examples embedded within docstrings as automated tests. This functionality ensures the accuracy of the examples and verifies that the code behaves as anticipated.

In the above example, the piece of code run and tested by doctest is below the
`Examples` tag. If the results are different from 8 or 9, an error will be generated. Importantly, this also give a clear example of how to use your code.

You can run doctest on your files outside of the GitHub CI if you want:

```Bash
python -m doctest my_module.py
```
If all doctests pass successfully, no output will be displayed. This signifies successful verification. If a doctest fails, an error message will be presented, indicating the specific example that failed and the discrepancy between the expected and actual output.

## Homogeneous and Correct Code with Ruff

[Ruff](https://github.com/astral-sh/ruff) functions as an automated code analysis tool, akin to a sophisticated proofreader and style advisor for your Python projects. It systematically examines your codebase, verifying compliance with a multitude of predefined rules to ensure adherence to established conventions and to identify potential sources of errors.

## Scope of Rules Under Consideration

We have pre-selected an ensemble of rules, but this may change in the future. These categories encompass a range of aspects related to code style, potential errors, and documentation standards:

* **E4:** Pertains to the appropriate use of blank lines to enhance code readability.
* **E7:** Addresses the consistent application of spacing around operators (e.g., +, -, =).
* **E9:** Identifies issues related to the presence or absence of a newline character at the end of a file.
* **F:** Flags common programming errors and constructs that may lead to unexpected behavior.
* **DOC:** Verifies the presence and quality of docstrings, which serve as explanatory documentation for code elements.
* **B9:** Detects potential bugs, particularly those arising from misunderstandings of how certain code constructs function.
* **B:** An additional set of rules focused on preventing the introduction of bugs.
* **E:** A general category encompassing style recommendations from the Pycodestyle tool.
* **W:** Highlights potential issues that, while not immediate errors, could lead to problems in the future.
* **ANN:** Encourages the use of type hints to improve code clarity regarding the expected data types of variables and function arguments.

Some of these rules will be blocking, as in you will need to make the changes yourself and re-commit the file, while others will be automatically fixed by ruff. In the latter case, you would still need to re-commit the impacted file.

# Pytest: Unit and Integration Testing

Testing ensures code correctness and reliability.

- **Unit Testing:** Verifies individual components (functions, methods) in isolation.
- **Integration Testing:** Validates the interaction between different components, like legos.

Pytest simplifies test creation and execution in Python.

## Writing Tests

1. **Test Files:** Create files named test_*.py.
2. **Test Functions:** Define functions named test_*().
3. **Assertions:** Use assert to verify expected outcomes.

Let's start with a simple suit of functions:

```python
# check_higher.py

def add(x : float, y : float):
  """Adds two numbers together."""
  return x + y

def check_if_higher(value : float):
  """Checks if a grade is passing (60 or above)."""
  return value >= 60

def format_answer(name : str, value : float):
  """Creates a feedback message based on the value."""
  if check_if_higher(value):
    return f"The {name} is higher than 60 {value}."
  else:
    return f"The {name} is lower than 60 {value}."
```

Unit tests are all about checking if individual parts (like our add function or check_if_higher function) work correctly on their own. We could create a `test_check_higher.py` file in `tests/unittests`:

```python
# test_check_higher.py
import pytest
from check_higher import add, check_if_higher

def test_add_positive_numbers():
  """Tests adding two positive numbers."""
  assert add(5, 3) == 8

def test_add_negative_numbers():
  """Tests adding two negative numbers."""
  assert add(-2, -4) == -6

def test_add_positive_and_negative():
  """Tests adding a positive and a negative number."""
  assert add(10, -5) == 5

def test_check_if_passed_passing_grade():
  """Tests if higher is correctly identified."""
  assert check_if_higher(75) is True

def test_check_if_passed_failing_grade():
  """Tests if lower is correctly identified."""
  assert check_if_higher(50) is False

def test_check_if_passed_borderline_grade():
  """Tests if the borderline passing value is correctly identified."""
  assert check_if_higher(60) is True
```

This can be tested with `pytest test_check_higher.py`.

Integration tests operate at a higher level. They typically are the example script that you would put on your repository. For instance, let's say that you have a library that helps to train a model for bird song detection. A good integration test would be a minimal example loading mockup data, instantiating the model, training for a few epochs and checking that the loss is actually decreasing. This ensures that when all the blocks of your library are connected together, they behave as expected.

