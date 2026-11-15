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

- **PyTorch** - The package uses PyTorch for tensor operations. Follow its
  security advisories at https://pytorch.org/
- Runtime dependencies are declared in `pyproject.toml`; development tools are
  optional extras and are not needed to use the package.

### Code Security

- The package's tensor operations do not make network requests or access files.
- The optional benchmark downloads its public dataset when it is not cached;
  see `benchmarks/real_task.py` before running it.
- Public tensor inputs and configuration values receive validation, but this
  does not replace validation in downstream applications.
- The package does not evaluate user-provided strings as code.

### Testing

- CI runs regression and property tests with a 95% package coverage threshold,
  plus Ruff and strict mypy checks. Coverage and tests do not establish training
  benefits or prove absence of defects.

## Best Practices for Users

### Installation
- Install from official PyPI: `pip install zeta-scaling-toolkit`
- Or from the verified GitHub repository
- Always use `pip install` from trusted sources only

### Usage
- Keep your PyTorch version up to date
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

Keep dependencies updated to receive security patches.

## Security Updates

Security patches will be released as soon as practical after vulnerabilities are verified and fixed. Users are encouraged to:
- Watch this repository for releases
- Subscribe to PyTorch security advisories
- Update regularly to the latest version

## Questions?

If you have security concerns or questions, please reach out to the maintainers privately rather than opening a public issue.
