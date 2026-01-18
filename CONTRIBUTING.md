## Contributing

Please open a pull request for any change.

### Rules

- Do not commit personal/local configuration (`config.json`) or any databases (`*.db`).
- Keep all paths in docs/examples generic (`/path/to/...`).
- Avoid committing generated files like `__pycache__/`.

### Checks

CI runs a lightweight syntax check (`python -m py_compile`) on `image_database.py` and `visualize_umap.py`.
