"""Rebuild a trained scene from its output directory.

Every script that consumes a checkpoint needs the same four steps: re-parse the
run's own config.yaml through the same parameter classes training used, extract
ModelParams, construct the scene, load the weights. This lived as three
near-copies across scripts/ that had already drifted apart -- two returned the
parsed args and one did not -- so a caller could not move between them.
"""

from pathlib import Path

from configs import (  # noqa: F401
    DatasetParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
from radfoam_model.scene import RadFoamScene


def load_model(checkpoint, device, model_file="model.pt"):
    """Return (model, args, dataset_args) for a run directory.

    `args` is the full re-parsed config, which the renderers need for
    resolution and background; most consumers want only the first and last.
    """
    import configargparse

    config = Path(checkpoint) / "config.yaml"
    parser = configargparse.ArgParser(default_config_files=[str(config)])
    parser.add_argument("-c", "--config", is_config_file=True)
    model_params = ModelParams(parser)
    PipelineParams(parser)
    OptimizationParams(parser)
    dataset_params = DatasetParams(parser)
    args = parser.parse_args(["-c", str(config)])

    model = RadFoamScene(args=model_params.extract(args), device=device)
    model.load_pt(str(Path(checkpoint) / model_file))
    return model, args, dataset_params.extract(args)
