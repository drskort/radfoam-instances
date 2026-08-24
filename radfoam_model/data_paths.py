"""Where the datasets live.

Nothing in this repo names a machine. Every root below is an environment
variable if set, else <repo>/data/<name>.

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

def resolve(name, env=None):
    """Root for `name`: $RADFOAM_<NAME> if set, else <repo>/data/<name>."""
    explicit = os.environ.get(env or f"RADFOAM_{name.upper()}")
    return Path(explicit) if explicit else DATA / name


def candidates(name, env=None):
    """Every plausible root for `name`, best first.

    Mask stores may be written by whichever machine ran the job, so consumers
    look through a list rather than trusting one location. Extra roots come from
    RADFOAM_<NAME>_EXTRA as a colon-separated list.
    """
    found = []
    explicit = os.environ.get(env or f"RADFOAM_{name.upper()}")
    if explicit:
        found.append(Path(explicit))
    found.append(DATA / name)
    extra = os.environ.get(f"RADFOAM_{name.upper()}_EXTRA", "")
    found.extend(Path(c) for c in extra.split(":") if c)
    return found


LERF_MASK_ROOT = resolve("lerf_mask")
LERF_OVS_ROOT = resolve("lerf_ovs")
SCANNETPP_ROOT = resolve("scannetpp")
MIPNERF360_ROOT = resolve("mipnerf360")
SAM_MASK_ROOTS = candidates("sam_masks")

# ScanNet++ ships data/, metadata/ and splits/ as siblings; the release root is
# therefore the parent of the data directory unless overridden separately.
# resolve() first: data/scannetpp is typically a symlink into the release, and
# metadata/ and splits/ are siblings of its target, not of the link.
_SNPP_RELEASE = Path(os.environ.get(
    "RADFOAM_SCANNETPP_RELEASE",
    (SCANNETPP_ROOT.resolve() if SCANNETPP_ROOT.exists() else SCANNETPP_ROOT).parent))
SCANNETPP_SPLITS = Path(os.environ.get("RADFOAM_SCANNETPP_SPLITS",
                                       _SNPP_RELEASE / "splits"))
SCANNETPP_META = Path(os.environ.get("RADFOAM_SCANNETPP_META",
                                     _SNPP_RELEASE / "metadata"))
