from __future__ import annotations

from pathlib import Path
import numpy as np
import yaml


def load_config(path):
    path = Path(path).expanduser()
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_dir"] = str(path.parent.resolve())
    return cfg


def ensure_dir(path):
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_root(cfg, model_name):
    return Path(cfg["models"][model_name]["root"]).expanduser()


def survey_output_dir(cfg, model_name):
    root = model_root(cfg, model_name)
    rel = cfg["models"][model_name].get("survey_output", "SURVEY_OUTPUT")
    return root / rel


def output_root(cfg):
    return ensure_dir(Path(cfg["output"]["directory"]).expanduser())


def figures_dir(cfg):
    return ensure_dir(output_root(cfg) / cfg["output"].get("figures_dir", "FIGURES"))


def numpy_dir(cfg):
    return ensure_dir(output_root(cfg) / cfg["output"].get("numpy_dir", "NUMPY"))


def mseed_dir(cfg):
    return ensure_dir(output_root(cfg) / cfg["output"].get("mseed_dir", "MSEED"))


def receiver_x(cfg):
    s = cfg["survey"]
    return float(s["first_receiver_x_m"]) + np.arange(int(s["n_receivers"])) * float(s["receiver_spacing_m"])


def shot_x(cfg):
    s = cfg["survey"]
    return float(s["first_shot_x_m"]) + np.arange(int(s["n_shots"])) * float(s["shot_spacing_m"])


def shot_dir_name(shot_number: int, source_x_m: float) -> str:
    xstr = f"{source_x_m:08.1f}".replace(".", "p")
    return f"shot_{shot_number:03d}_xs{xstr}"


def find_shot_dir(survey_dir, shot_number: int, source_x_m: float | None = None):
    survey_dir = Path(survey_dir).expanduser()
    patterns = [
        f"shot_{shot_number:03d}_*",
        f"shot_{shot_number:03d}",
        f"shot{shot_number:03d}_*",
        f"*{shot_number:03d}*",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(sorted(survey_dir.glob(pat)))
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No shot directory found for shot {shot_number} in {survey_dir}")
    if len(candidates) > 1 and source_x_m is not None:
        token = f"{source_x_m:.1f}".replace(".", "p")
        for c in candidates:
            if token in c.name:
                return c
    return candidates[0]


def read_source_function(path):
    path = Path(path).expanduser()
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        y = arr
        t = np.arange(len(y), dtype=float)
    else:
        t = arr[:, 0]
        y = arr[:, 1]
    return t, y
