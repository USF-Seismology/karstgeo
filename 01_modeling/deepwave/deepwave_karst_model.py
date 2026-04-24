# deepwave_karst_model.py
# (shortened header for retry)
import argparse, yaml, os
from pathlib import Path
import numpy as np
import torch
import deepwave
from deepwave import scalar
from obspy import Stream, Trace, UTCDateTime

def load_config(p):
    with open(p) as f: return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(os.cpu_count())
    device = torch.device("cpu")

    dx = cfg["model"]["dx"]
    x = np.arange(cfg["model"]["x_min"], cfg["model"]["x_max"]+dx, dx)
    z = np.arange(0, cfg["model"]["z_max"]+dx, dx)

    # simple velocity
    v = np.ones((len(z), len(x)), dtype=np.float32) * cfg["velocity"]["background"]

    nodes = np.arange(cfg["survey"]["n_nodes"]) * cfg["survey"]["node_spacing"]
    shots = np.unique(np.concatenate([
        cfg["survey"]["offend_shots"],
        nodes,
        cfg["survey"]["far_shots"]
    ]))

    outdir = Path(cfg["output"]["dir"])
    outdir.mkdir(exist_ok=True)

    for i, sx in enumerate(shots):
        print("Shot", i, sx)

        # minimal run (placeholder safe)
        data = np.zeros((len(nodes), cfg["model"]["nt"]), dtype=np.float32)

        st = Stream()
        t0 = UTCDateTime(2026,1,1)
        for j in range(len(nodes)):
            tr = Trace(data=data[j])
            tr.stats.delta = cfg["model"]["dt"]
            tr.stats.starttime = t0
            st.append(tr)

        st.write(str(outdir / f"shot_{i:03d}.mseed"), format="MSEED")

if __name__ == "__main__":
    main()
