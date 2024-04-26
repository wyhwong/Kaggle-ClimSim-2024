# Clim-Sim 2024

## Installation

```bash
# Install poetry if needed
# Do not forget to set up virtual environment before this
pip3 install poetry

# Install dependencies
make install
```

## Development

> [!CAUTION]
> After `make update`, please run test cases to check if any function is deprecated due to packge updates.

```bash
# Run static code analysis
make static-code-analysis

# Run test cases in poetry environment
make test

# Format code with black, isort
make format

# Update dependencies
make update
```
