# Contributing to ThinkRouter

Thank you for your interest in contributing. ThinkRouter is an open-source project and all contributions — bug reports, feature requests, documentation improvements, and code — are welcome.

---

## Getting started

### 1. Fork and clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/thinkrouter.git
cd thinkrouter
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows
```

### 3. Install in editable mode with dev dependencies

```bash
pip install -e ".[dev]"
```

### 4. Verify the test suite passes

```bash
pytest tests/ -v
```

All 69+ tests should pass before you make any changes.

---

## Making changes

### Branch naming

```
feature/short-description
fix/short-description
docs/short-description
```

### Code style

- We use `ruff` for linting. Run `ruff check thinkrouter/` before committing.
- Line length: 100 characters.
- Type hints are required on all public functions.
- Docstrings follow the NumPy convention (Parameters / Returns sections).

### Adding a new feature

1. Write the code in the appropriate module.
2. Add or update tests in `tests/test_thinkrouter.py`.
3. Verify all tests pass: `pytest tests/ -v`
4. Update the docstring and README if the public API changes.

### Adding a new classifier backend

1. Subclass `BaseClassifier` in `thinkrouter/classifier.py`.
2. Implement `predict(self, query: str) -> ClassifierResult`.
3. Register it in the `get_classifier()` factory.
4. Add tests in the `TestGetClassifier` class.
5. Document the new backend in the README under "Classifier backends".

---

## Submitting a pull request

1. Push your branch to your fork.
2. Open a PR against `thinkrouter/thinkrouter:main`.
3. Fill in the PR template (what changed, why, how to test).
4. CI must pass before review.

---

## Reporting bugs

Open an issue at https://github.com/thinkrouter/thinkrouter/issues with:

- Python version and OS
- ThinkRouter version (`python -c "import thinkrouter; print(thinkrouter.__version__)"`)
- Minimal reproducible example
- Full error traceback

---

## Roadmap and open issues

See the Issues tab for features marked `help wanted` and `good first issue`.
Current priorities:

- [ ] Publish DistilBERT model to HuggingFace Hub
- [ ] Multi-domain training (MMLU, HumanEval, ARC-Challenge)
- [ ] Async support (`asyncio` compatible `achat()` and `astream()`)
- [ ] Continuous budget regression (replace discrete 3-tier with scalar prediction)
- [ ] Hosted proxy (api.thinkrouter.ai)

---

## License

By contributing, you agree your contributions will be licensed under the MIT License.
