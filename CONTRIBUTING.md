# Contributing to Pet Face Classification

Thank you for your interest in contributing! All contributions are welcome, from bug fixes and documentation improvements to new features and experiments.

## Getting Started

1. **Fork** the repository and clone your fork locally.
2. Create a **feature branch**: `git checkout -b feature/your-feature-name`
3. Install dependencies: `pip install -e .[dev]`
4. Make your changes and commit with a clear message.
5. **Push** your branch and open a **Pull Request** against `main`.

## Development Setup

```bash
git clone https://github.com/<your-username>/classification_pet_faces.git
cd classification_pet_faces
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Add docstrings to any new functions and classes.
- Keep functions focused — single responsibility.
- All paths should use `pathlib.Path`, not string concatenation.

## Reporting Bugs

Please open a [GitHub Issue](../../issues) with:
- A clear description of the bug
- Steps to reproduce it
- Your Python and TensorFlow version (`python --version`, `python -c "import tensorflow; print(tensorflow.__version__)"`)

## Suggesting Enhancements

Open an issue with the `enhancement` label describing:
- What you'd like to see
- Why it would be useful
- Any implementation ideas you have

## Ideas for Contributions

- Add fine-tuning support (unfreeze ResNet50 layers progressively)
- Add Grad-CAM visualizations for model interpretability
- Add a `predict.py` inference script for new images
- Write unit tests in `tests/`
- Add support for other pretrained backbones (EfficientNet, MobileNet)
- Add a `Makefile` for common commands

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
