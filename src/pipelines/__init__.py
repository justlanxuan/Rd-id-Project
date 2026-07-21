"""Legacy pipeline package.

Use `python -m src.pipeline` or `src.pipeline.run_pipeline` for new code.
"""

from src.pipelines.full_pipeline import FullPipeline

__all__ = ["FullPipeline"]
