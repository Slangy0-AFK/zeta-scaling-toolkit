# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in the Zeta Scaling Toolkit, please **do not** open a public issue. Instead, please email your findings to the maintainers privately.

### How to Report

1. **Do not disclose the vulnerability publicly** until we've had time to address it
2. **Include details:**
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if you have one)

3. **Expected timeline:**
   - We aim to acknowledge reports within 48 hours
   - Critical issues will be prioritized for immediate patching

## Security Practices

### Dependencies

- **PyTorch** - We rely on PyTorch for all tensor operations. Follow PyTorch's security advisories at https://pytorch.org/
- **NumPy** - Numerical operations are delegated to NumPy. See https://numpy.org/security for updates
- All dependencies are pinned to known-good versions in `pyproject.toml`

### Code Security

- **No external API calls** - This library does not make HTTP requests or contact external services
- **No file I/O** - The toolkit operates entirely on in-memory tensors
- **Input validation** - All inputs are validated to prevent unexpected tensor operations
- **No arbitrary code execution** - The toolkit does not eval or exec user input

### Testing

- **100% test coverage** - All functionality is covered by automated tests
- **Type checking** - Optional static type hints with mypy for type safety
- **Regular CI/CD** - Tests run on every commit via GitHub Actions

## Best Practices for Users

### Installation
- Install from official PyPI: `pip install zeta-scaling-toolkit`
- Or from the verified GitHub repository
- Always use `pip install` from trusted sources only

### Usage
- Keep your PyTorch and NumPy versions up to date
- Follow PyTorch's security guidelines when building production models
- The toolkit is experimental - evaluate against benchmarks before production deployment

### Deployment
- This is a mathematical library for AI scaling - use it as one component of your ML pipeline
- Implement proper data validation and model testing in your own code
- Follow your organization's ML security policies

## Known Limitations

- **Experimental status** - This is research code and should be treated as such
- **No cryptography** - This library does not provide encryption or security primitives
- **No authentication** - The toolkit has no user authentication or access control
- **Python security** - Depends on Python interpreter security; use Python 3.9+

## Version Support

- Python 3.9+ is supported
- PyTorch 2.0.0+
- NumPy 1.21.0+

Always keep your dependencies updated to receive security patches.

## Security Updates

Security patches will be released as soon as practical after vulnerabilities are verified and fixed. Users are encouraged to:
- Watch this repository for releases
- Subscribe to PyTorch and NumPy security advisories
- Update regularly to the latest version

## Questions?

If you have security concerns or questions, please reach out to the maintainers privately rather than opening a public issue.
