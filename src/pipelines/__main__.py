"""Compatibility wrapper for the old `python -m src.pipelines` entrypoint."""

from src.pipeline import main


if __name__ == "__main__":
    main()
