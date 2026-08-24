"""Where the datasets live.

Nothing in this repo should name a machine. Every root below is resolved in
three steps: an environment variable if set, else <repo>/data/<name>, else the
paths of the cluster this was developed on -- tried last and only if they exist,
so a clone elsewhere never sees them.

The intended setup is symlinks, which is why the defaults are inside the repo:

    mkdir -p data
    ln -s /path/to/lerf_mask  data/lerf_mask
    ln -s /path/to/scannetpp  data/scannetpp

`data/` is gitignored. See the README's dataset section for what each root is
expected to contain.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("RADFOAM_DATA", REPO / "data"))

# Kept only so the original cluster keeps working without a setup step. They are
# checked for existence before use, so they are inert anywhere else.
_LEGACY = {
    "lerf_mask": ["/nodes/host/work/user/lerf_mask"],
    "lerf_ovs": ["/nodes/host/work/user/lerf_ovs"],
    "scannetpp": ["/shared/scannetpp/data"],
    "sam_masks": ["/nodes/host/work/user/sam_masks",
                  "/work/user/sam_masks"],
    "mipnerf360": ["/shared/user/datasets"],
}


def resolve(name, env=None):
    """Root for `name`: $env, else data/<name>, else a legacy path that exists."""
    explicit = os.environ.get(env or f"RADFOAM_{name.upper()}")
    if explicit:
        return Path(explicit)
    default = DATA / name
    if default.exists():
        return default
    for candidate in _LEGACY.get(name, ()):
        if Path(candidate).exists():
            return Path(candidate)
    return default


def candidates(name, env=None):
    """Every plausible root for `name`, best first.

    Mask stores are written by whichever machine ran the job, so consumers look
    through a list rather than trusting one location.
    """
    found = []
    explicit = os.environ.get(env or f"RADFOAM_{name.upper()}")
    if explicit:
        found.append(Path(explicit))
    found.append(DATA / name)
    found.extend(Path(c) for c in _LEGACY.get(name, ()))
    return found


LERF_MASK_ROOT = resolve("lerf_mask")
LERF_OVS_ROOT = resolve("lerf_ovs")
SCANNETPP_ROOT = resolve("scannetpp")
MIPNERF360_ROOT = resolve("mipnerf360")
SAM_MASK_ROOTS = candidates("sam_masks")

# ScanNet++ ships data/, metadata/ and splits/ as siblings; the release root is
# therefore the parent of the data directory unless overridden separately.
_SNPP_RELEASE = Path(os.environ.get("RADFOAM_SCANNETPP_RELEASE",
                                    SCANNETPP_ROOT.parent))
SCANNETPP_SPLITS = Path(os.environ.get("RADFOAM_SCANNETPP_SPLITS",
                                       _SNPP_RELEASE / "splits"))
SCANNETPP_META = Path(os.environ.get("RADFOAM_SCANNETPP_META",
                                     _SNPP_RELEASE / "metadata"))
