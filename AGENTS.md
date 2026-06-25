# Project Guidelines

**Rule number 0:** Always use the simplest correct (conventional standard) solution to solve a problem. Avoid over-engineering and unnecessary complexity. But also avoid "dumb" solutions that are not maintainable or scalable. Always consider the long-term implications of your code and strive for a balance between simplicity and functionality.
**Rule number 1:** Always write clear and concise code. Avoid unnecessary complexity.
**Rule number 2:** Follow consistent naming conventions for variables, functions, and classes.
**Rule number 3:** Document your code with docstrings to explain the purpose and functionality.
**Rule number 4:** Write unit tests to ensure your code works as expected and to catch potential bugs early.
**Rule number 5:** All code here must be device agnostic, meaning it works on GPU (Cuda) or CPU and is optimized on both scenarios.
**Rule number 7:** Avoid the usage of "try/except" blocks for control flow. Use them only for handling exceptions that are truly exceptional. And even then, crash, dont fallback. Only exception is when handling cross platform imports or specific device code that cannot be handled by "if/else" for some reason. And if what you need is only to check if a condition is match and crash if dont, use "assert" instead of "try/except" blocks.
**Rule number 8:** Always run "ruff check --fix" and "uv run pytest" to detect issus and prevent regressions before committing code. If you are not sure about the issues, ask for help.
**Rule number 9:** Always use "ruff format" to format your code before committing.
