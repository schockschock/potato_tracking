import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_ENV_LOADED = False


def _ensure_dotenv_loaded() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = Path(os.getenv("POTATO_ENV", ""))
    if env_path.is_file():
        load_dotenv(env_path)
        _ENV_LOADED = True
        return
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for p in candidates:
        if p.is_file():
            load_dotenv(p)
            _ENV_LOADED = True
            return
    _ENV_LOADED = True


def get_data_dir() -> Optional[Path]:
    _ensure_dotenv_loaded()
    val = os.getenv("DATA_DIR", "")
    return Path(val) if val else None


def get_output_dir() -> Optional[Path]:
    _ensure_dotenv_loaded()
    val = os.getenv("OUTPUT_DIR", "")
    return Path(val) if val else None


@dataclass
class PipelineParams:
    threshold: int = 20
    morph_kernel_size: int = 7
    min_area: int = 500
    grabcut_rect_proportion: tuple[float, float, float, float] = (
        0.05, 0.40, 0.90, 0.55,
    )
    grabcut_iterations: int = 5
