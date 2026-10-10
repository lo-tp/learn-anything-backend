"""One-shot analysis scripts (run against the database, not imported at runtime).

The package exists so the tests can import a script module by name
(``scripts.classify_slide_failures``) and be type-checked doing it; the scripts
themselves are still run as files: ``python scripts/<name>.py``.
"""
