# Contributing to Zeta Scaling Toolkit

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful and constructive in all interactions. We're building a community of researchers and developers working on AI scaling.

## How to Contribute

### Reporting Bugs

1. **Check existing issues** - Search to see if the bug has been reported
2. **Create a detailed report** with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs. actual behavior
   - Your environment (Python version, PyTorch version, OS)
   - Code snippet if possible

### Suggesting Features

1. **Open an issue** describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Why this would be valuable to the project
   - Any relevant research or references

### Submitting Pull Requests

#### Prerequisites
- Python 3.9+
- PyTorch 2.0.0+
- Familiarity with the W(T) = 1 + A₀/ln(T) framework

#### Setup for Development
```bash
# Clone the repository
git clone https://github.com/Slangy0-AFK/zeta-scaling-toolkit.git
cd zeta-scaling-toolkit

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,test]"
```

#### Code Style

We follow PEP 8 with the following tools:

```bash
# Format code with Black
black zeta_scaling_toolkit/ tests/

# Lint with Ruff
ruff check zeta_scaling_toolkit/ tests/

# Type check with mypy
mypy zeta_scaling_toolkit/

# All together
black zeta_scaling_toolkit/ tests/ && ruff check zeta_scaling_toolkit/ tests/ --fix && mypy zeta_scaling_toolkit/
```

**Style Rules:**
- Line length: 100 characters (enforced by Black)
- Target Python: 3.9+ (enforced by Ruff)
- 4-space indentation
- Docstrings on all public functions
- Type hints encouraged

#### Testing

Before submitting a PR, ensure:

```bash
# Run all tests
pytest tests/ -v --cov=zeta_scaling_toolkit

# Coverage must be 100%
pytest tests/ --cov=zeta_scaling_toolkit --cov-fail-under=100
```

**Test Requirements:**
- Add tests for any new functionality
- Add tests for bug fixes
- Maintain or improve code coverage (target: 100%)
- Tests must pass on Python 3.9, 3.10, 3.11, 3.12

#### Pull Request Process

1. **Fork the repository** on GitHub

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clear, descriptive commit messages
   - Reference related issues in commit messages
   - Keep commits focused and atomic

4. **Test thoroughly**
   ```bash
   pytest tests/ -v --cov=zeta_scaling_toolkit
   black zeta_scaling_toolkit/ tests/
   ruff check zeta_scaling_toolkit/ tests/ --fix
   mypy zeta_scaling_toolkit/
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Use a clear, descriptive title
   - Reference related issues (e.g., "Fixes #123")
   - Describe your changes and why they're needed
   - Link to any relevant research or documentation

7. **Address review feedback**
   - Respond to code review comments
   - Make requested changes
   - Push updates to the same branch

8. **Merge**
   - Squash commits if requested
   - Maintainers will merge when ready

## Development Workflow

### File Structure
```
zeta_scaling_toolkit/
├── __init__.py           # Public API exports
└── core.py              # Core implementation
tests/
└── test_zeta_toolkit.py # Test suite
```

### Adding New Features

1. **Implement in `core.py`**
   - Follow existing code style
   - Include comprehensive docstrings
   - Add type hints

2. **Export in `__init__.py`**
   ```python
   from .core import YourNewClass, your_new_function
   
   __all__ = [
       "YourNewClass",
       "your_new_function",
       # ... existing exports
   ]
   ```

3. **Test in `tests/test_zeta_toolkit.py`**
   - Create a new test class
   - Test all public methods
   - Test edge cases
   - Aim for 100% coverage

### Documentation

- **Docstrings**: Use Google-style docstrings
  ```python
  def my_function(x: Tensor) -> Tensor:
      """Brief description.
      
      Longer description explaining the mathematical or practical
      motivation for this function.
      
      Args:
          x: Input tensor of shape (batch, seq_len, dim)
          
      Returns:
          Output tensor of shape (batch, seq_len, dim)
          
      Raises:
          ValueError: If dimensions are incompatible
          
      Example:
          >>> x = torch.randn(2, 128, 64)
          >>> output = my_function(x)
          >>> output.shape
          torch.Size([2, 128, 64])
      """
  ```

- **Comments**: Only comment non-obvious logic
  - Avoid stating the obvious
  - Explain the "why", not the "what"

- **README**: Update if your changes affect user-facing features

### Mathematical Notation

When explaining the W(T) framework or related math:
- Use clear mathematical notation
- Explain assumptions
- Cite relevant research
- Include references in docstrings

Example:
```python
def asymptotic_floor(t: float, a0: float = 100 * math.pi) -> float:
    """Compute W(T) = 1 + A₀/ln(T).
    
    This provides an asymptotic lower bound that prevents
    scaling factors from collapsing to zero.
    
    References:
        [1] Your paper/research citation
    """
```

## Review Process

### Expectations
- Be respectful and constructive
- Respond to feedback promptly
- Ask questions if clarification is needed
- Thank reviewers for their time

### What Reviewers Look For
- [ ] Code follows style guidelines
- [ ] Tests added/updated with 100% coverage
- [ ] Documentation is clear and complete
- [ ] Changes align with project scope
- [ ] No breaking changes without justification
- [ ] Performance implications considered

## Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

1. **Patch** (e.g., 0.1.1): Bug fixes
2. **Minor** (e.g., 0.2.0): New features, backward compatible
3. **Major** (e.g., 1.0.0): Breaking changes

Maintainers handle releases and PyPI publishing.

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Documentation**: See README.md and TESTING.md

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Attribution

Contributors will be recognized in:
- Release notes
- CONTRIBUTORS.md (if we create one)
- GitHub contributor graph

## Questions?

Please open an issue with the `question` label or check existing discussions before asking.

Thank you for contributing! 🙏
