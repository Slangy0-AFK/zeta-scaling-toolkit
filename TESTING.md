# Testing Guide for Users

This guide explains how to run and write tests for the Zeta Scaling Toolkit.

## Quick Start

### Prerequisites
```bash
# Make sure you have Python 3.9 or higher
python --version
```

### Install for Testing
```bash
# Clone the repository
git clone https://github.com/Slangy0-AFK/zeta-scaling-toolkit.git
cd zeta-scaling-toolkit

# Install the toolkit with test dependencies
pip install -e ".[test]"
```

### Run All Tests
```bash
pytest tests/ -v
```

For a separate smoke check, run `python zeta_scaling_toolkit.py`. These tests
check formulas and basic behavior; they do not establish training benefits.

## Viewing Test Coverage

### Generate Coverage Report
```bash
pytest tests/ -v --cov=zeta_scaling_toolkit --cov-report=html
```

This generates:
- **Terminal report** - Summary printed to console
- **HTML report** - Open `htmlcov/index.html` in your browser for detailed coverage

### Coverage Goals
- Target: **100% code coverage**
- All functions and branches should be tested
- All edge cases should be validated

## Understanding the Tests

### Test Structure
```
tests/
├── test_zeta_toolkit.py          # Main test suite
└── ... (additional test files)
```

### Test Groups

#### 1. NonCollapsingAttention Tests
- Checks the attention temperature formula and invalid sequence length

#### 2. AsymptoticLRScheduler Tests
- Validates learning rate decay schedule
- Verifies the starting rate, logarithmic decay, and configured floor

#### 3. Scale-Invariant Vision Coords Tests
- Tests coordinate scaling for different image sizes
- Validates polar coordinate transformation
- Checks radius and output shape

#### 4. Linear Initialization Tests
```python
test_linear_uses_kaiming_variance
```
- Checks Kaiming initialization and zero bias

## Running Specific Tests

### Run Attention Tests
```bash
pytest tests/test_zeta_toolkit.py -k attention -v
```

### Run a Specific Test
```bash
pytest tests/test_zeta_toolkit.py::test_attention_uses_mild_temperature -v
```

### Run Tests Matching a Pattern
```bash
pytest tests/ -k "attention" -v
```

## Writing Your Own Tests

### Basic Test Template
```python
import pytest
import torch
from zeta_scaling_toolkit import NonCollapsingAttention

class TestMyFeature:
    def test_basic_functionality(self):
        """Test that the feature works correctly."""
        # Setup
        component = NonCollapsingAttention(d_k=64)
        Q = torch.randn(2, 128, 64)
        K = torch.randn(2, 128, 64)
        V = torch.randn(2, 128, 64)
        
        # Execute
        output = component(Q, K, V)
        
        # Assert
        assert output.shape == (2, 128, 64)
        assert not torch.isnan(output).any()
```

### Adding Tests to the Project
1. Add test methods to `tests/test_zeta_toolkit.py`
2. Follow naming convention: `test_<feature>_<scenario>`
3. Run tests to verify: `pytest tests/test_zeta_toolkit.py -v`

## Continuous Integration

The project uses GitHub Actions for automated testing. Tests run on:
- Every commit to main branches
- Every pull request
- Python 3.9, 3.10, 3.11, 3.12

Check `.github/workflows/test.yml` for CI configuration.

## Troubleshooting

### Test Failures

**If tests fail, check:**

1. **Python version**
   ```bash
   python --version  # Should be 3.9+
   ```

2. **Dependencies installed**
   ```bash
   pip list | grep -E "torch|numpy|pytest"
   ```

3. **CUDA availability (optional)**
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Run with verbose output**
   ```bash
   pytest tests/ -v --tb=long
   ```

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'zeta_scaling_toolkit'` | Run `pip install -e .` |
| `pytest: command not found` | Run `pip install pytest pytest-cov` |
| Tests hang or are very slow | Reduce batch sizes in tests or run on GPU |
| CUDA out of memory | Tests automatically use CPU if GPU unavailable |

## Performance Testing

### Benchmark Attention Performance
```python
import torch
import time
from zeta_scaling_toolkit import NonCollapsingAttention

attn = NonCollapsingAttention(d_k=64)
Q = torch.randn(1, 1024, 64)  # Long sequence
K = torch.randn(1, 1024, 64)
V = torch.randn(1, 1024, 64)

start = time.time()
output = attn(Q, K, V)
end = time.time()

print(f"Time: {(end-start)*1000:.2f}ms")
```

## Integration Testing

When integrating the toolkit into your own project:

```python
# test_my_model.py
import torch
from zeta_scaling_toolkit import (
    NonCollapsingAttention,
    AsymptoticLRScheduler,
    make_corrected_linear
)

def test_model_with_zeta_components():
    """Test your model with Zeta Scaling components."""
    
    # Build model with toolkit components
    model = MyModel(
        attention=NonCollapsingAttention(d_k=64),
        linear=make_corrected_linear(1024, 4096)
    )
    
    # Test forward pass
    x = torch.randn(2, 128, 1024)
    output = model(x)
    
    assert output.shape == (2, 128, 4096)
    
    # Test with optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = AsymptoticLRScheduler(optimizer)
    
    # Training loop
    for epoch in range(10):
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    
    # Verify LR never reaches zero
    assert scheduler.get_last_lr()[0] > 0
```

## Test Maintenance

### Updating Tests
- Keep tests synchronized with code changes
- Add tests for new features before merging
- Update tests if behavior changes
- Maintain 100% coverage

### Test Review Checklist
- [ ] Tests pass locally: `pytest tests/ -v`
- [ ] Coverage is 100%: `pytest --cov`
- [ ] No warnings or errors
- [ ] Tests are deterministic (no random failures)
- [ ] Tests document expected behavior

## Additional Resources

- **PyTorch Testing Guide**: https://pytorch.org/docs/stable/testing.html
- **Pytest Documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/

## Contributing Tests

When submitting pull requests, include:
1. Tests for new functionality
2. Tests for bug fixes
3. Updated or new integration tests
4. 100% code coverage for your changes

See CONTRIBUTING.md for more details on submitting contributions.
