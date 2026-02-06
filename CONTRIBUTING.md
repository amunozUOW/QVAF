# Contributing to QVAF

Thank you for your interest in contributing to the Quiz Vulnerability Assessment Framework! This document provides guidelines for contributing to the project.

## Ways to Contribute

### Reporting Issues

If you encounter a bug or have a feature suggestion:

1. **Check existing issues** to avoid duplicates
2. **Use the issue template** when creating a new issue
3. **Provide context**: Include your operating system, Python version, and steps to reproduce

### Documentation Improvements

Documentation contributions are highly valued:

- Fix typos or unclear explanations
- Add examples for different disciplines
- Translate documentation
- Improve installation instructions for specific platforms

### Code Contributions

We welcome code contributions for:

- Bug fixes
- New LMS platform support (Canvas, Blackboard, etc.)
- Additional LLM provider integrations
- Performance improvements
- Test coverage expansion

---

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- Ollama installed locally
- Google Chrome

### Setting Up Your Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/qvaf.git
cd qvaf

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install Playwright browsers
python -m playwright install chromium

# Pull required Ollama models
ollama pull llama3:8b
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=. --cov-report=html

# Run specific test file
python -m pytest tests/test_reform_agent.py
```

### Code Style

We use:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Before submitting:

```bash
black .
isort .
flake8 .
```

---

## Contribution Process

### For Small Changes (typos, documentation)

1. Fork the repository
2. Make your changes
3. Submit a pull request

### For Larger Changes

1. **Open an issue first** to discuss the proposed change
2. Wait for feedback before investing significant effort
3. Fork the repository
4. Create a feature branch: `git checkout -b feature/your-feature-name`
5. Make your changes with clear, atomic commits
6. Add tests for new functionality
7. Update documentation as needed
8. Submit a pull request

### Pull Request Guidelines

- **Clear title**: Summarise the change in ~50 characters
- **Description**: Explain what and why (not how—the code shows how)
- **Link issues**: Reference any related issues
- **Small scope**: One feature or fix per PR
- **Tests pass**: Ensure all tests pass before requesting review

---

## Areas Where Help Is Particularly Welcome

### High Priority

- **LMS Support**: Adding support for Canvas, Blackboard, or other learning management systems
- **Test Coverage**: Expanding unit and integration tests
- **Documentation**: Improving clarity, adding examples, translations

### Medium Priority

- **Performance**: Optimising scan speed, reducing memory usage
- **UI/UX**: Improving the Streamlit interface
- **Accessibility**: Ensuring reports are accessible

### Research Contributions

- **Empirical validation**: Studies testing QVAF's effectiveness
- **Cross-disciplinary testing**: Applying the tool across different fields
- **Taxonomy refinement**: Improving cognitive demand classification

---

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what's best for the community

### Unacceptable Behaviour

- Harassment or discrimination
- Trolling or insulting comments
- Public or private attacks
- Publishing others' private information

### Enforcement

Instances of unacceptable behaviour may be reported to amunoz@uow.edu.au. All complaints will be reviewed and investigated.

---

## Recognition

Contributors will be acknowledged in:

- The README.md contributors section
- Release notes for significant contributions
- Academic publications arising from the project (where appropriate)

---

## Questions?

- **General questions**: Open a Discussion on GitHub
- **Bug reports**: Open an Issue (preferred method for tracking)
- **Security concerns**: Email amunoz@uow.edu.au directly

Thank you for contributing to better assessment design in the age of AI!
