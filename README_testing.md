# LLM Annotator - Project Structure and Testing Guide

This document outlines the recommended project structure and explains how to set up and run the unit tests for the `LLMAnnotator` class.

## Project Structure

Organize your project with the following directory structure:

```
masterblaster-classifier-trainer/
├── llm_annotator.py        # The optimized LLM Annotator class
├── free_al.py              # Your existing free_al implementation
├── main.py                 # Your main application entry point
├── tests/                  # Test directory
│   ├── __init__.py
│   ├── test_llm_annotator.py   # Unit tests for LLM Annotator
│   └── test_free_al.py         # Tests for free_al (if needed)
├── test_requirements.txt   # Test dependencies
├── pytest.ini              # Pytest configuration
└── run_tests.sh            # Test runner script
```

## Setting Up the Test Environment

1. Create a virtual environment (if not already using one):
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```bash
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install the test requirements:
   ```bash
   pip install -r test_requirements.txt
   ```

## Running the Tests

You can run the tests using the provided shell script:

```bash
chmod +x run_tests.sh
./run_tests.sh
```

Or run pytest directly:

```bash
python -m pytest tests/ -v
```

To generate a coverage report:

```bash
python -m pytest tests/ -v --cov=llm_annotator --cov-report=term --cov-report=html
```

This will generate an HTML coverage report in the `htmlcov/` directory.

## Test Structure

The tests are organized as follows:

1. **Basic Initialization Tests**: Verify that the class initializes correctly with various parameters.
2. **Thread Initialization Tests**: Test the thread initialization with and without demonstrations.
3. **API Interaction Tests**: Test calls to the Anthropic API with proper mocking.
4. **Label Matching Tests**: Test the Levenshtein distance-based label matching.
5. **Batch Processing Tests**: Test the batch annotation functionality.
6. **Retry Mechanism Tests**: Test the API retry mechanism for handling errors.

## Mock Strategy

The tests use Python's `unittest.mock` module to mock the Anthropic API:

1. The `anthropic.Anthropic` class is mocked to prevent actual API calls.
2. The `messages.create` method is mocked to return predefined responses.
3. Input prompts are mocked to avoid requiring user input during tests.

This approach allows testing the functionality without making actual API calls, which would be slow and costly.

## Adding New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py` for test files and `test_*` for test methods.
2. Use the provided mocking patterns to avoid actual API calls.
3. Add appropriate assertions to verify the expected behavior.
4. Run the tests to ensure they pass and maintain coverage.

## Interpreting Test Results

After running the tests, you'll see:

- A summary of test results (passed, failed, skipped)
- Code coverage statistics
- Location of any test failures

For any failing tests, examine the error message and stack trace to understand what went wrong and make the necessary fixes to your code.