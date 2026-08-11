import configargparse
import os
from argparse import Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(
        self, parser: configargparse.ArgParser, name: str, fill_none=False
    ):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            t = type(value)
            value = value if not fill_none else None
            if t == bool:
                group.add_argument(
                    "--" + key, default=value, action="store_true"
                )
            elif t == list:
                group.add_argument(
                    "--" + key,
                    nargs="+",
                    type=type(value[0]),
                    default=value,
                    help=f"List of {type(value[0]).__name__}",
                )
            else:
                group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class PipelineParams(ParamGroup):

    def __init__(self, parser):
        self.iterations = 20_000
        self.densify_from = 2_000
        self.densify_until = 11_000
        self.densify_factor = 1.15
        self.white_background = True
        self.quantile_weight = 1e-4
        # Contrastive instance features. instance_weight = 0 disables the loss
        # entirely; instance_guided_geometry lets it also shape density.
        # Save model.pt every N iterations, plus a numbered snapshot so
        # intermediate scenes can be rendered. 0 disables.
        self.checkpoint_every = 2_000
        self.instance_weight = 0.1
        self.variance_weight = 0.5
        # Iteration at which instance_guided_geometry starts taking effect.
        # Before it, features are still noise and moving geometry with them
        # only destabilises the triangulation.
        self.instance_geometry_from = 2_000
        # Per-level multipliers on the contrastive loss, one per entry of
        # instance_levels. See multi_level_instance_loss for why they are not
        # all 1.0 by default in every experiment.
        self.instance_level_weights = [1.0, 1.0, 1.0]
        # Potts prior on the occupancy field: commit every cell to solid or
        # empty, and minimise the interface between them. Both zero disables it.
        # See docs/specs/2026-08-11-occupancy-potts-prior-design.md.
        # Continue from an existing checkpoint rather than initialising from
        # COLMAP points. Requires densification off; see train.py.
        self.resume_from = ""
        # Global step to continue from. Learning rates, freeze_points and
        # occupancy_from are all functions of the step, so a resumed run that
        # restarted the counter at 0 would slam a converged model with the
        # initial learning rate.
        self.start_iteration = 0
        # Print a timestamped line around every triangulation
        # rebuild. The occupancy prior hangs only when sites are
        # free to move, and rebuild() is the one call on that
        # path the frozen probes never make.
        self.debug_triangulation = False
        self.occupancy_bin_weight = 0.0
        self.occupancy_tv_weight = 0.0
        self.occupancy_penalty = "entropy"
        self.occupancy_from = 0
        # Edges sampled per step for the interface term. 0 uses all ~15M, which
        # fits but costs; a sample is an unbiased estimate of the same mean.
        self.occupancy_edge_sample = 2_000_000
        self.instance_gamma = 1.0
        self.instance_pos_weight = 1.0
        self.instance_neg_weight = 1.0
        self.experiment_name = ""
        self.debug = False
        self.viewer = False
        super().__init__(parser, "Setting Pipeline parameters")


class ModelParams(ParamGroup):

    def __init__(self, parser):
        self.sh_degree = 3
        self.feat_dim = 16
        self.instance_guided_geometry = False
        self.init_points = 131_072
        self.final_points = 2_097_152
        self.activation_scale = 1.0
        self.device = "cuda"
        super().__init__(parser, "Setting Model parameters")


class OptimizationParams(ParamGroup):

    def __init__(self, parser):
        self.points_lr_init = 2e-4
        self.points_lr_final = 5e-6
        self.density_lr_init = 1e-1
        self.density_lr_final = 1e-2
        self.attributes_lr_init = 5e-3
        self.attributes_lr_final = 5e-4
        self.sh_factor = 0.1
        self.freeze_points = 18_000
        super().__init__(parser, "Setting Optimization parameters")


class DatasetParams(ParamGroup):

    def __init__(self, parser):
        self.dataset = "colmap"
        self.data_path = "data/mipnerf360"
        self.scene = "bonsai"
        self.patch_based = False
        self.downsample = [4, 2, 1]
        self.downsample_iterations = [0, 150, 500]
        # Which SAM granularity levels to load as supervision. Level 1 is the
        # degenerate one (4.1 masks/frame covering 44% of pixels); dropping it
        # is the ablation, down-weighting it via instance_level_weights is the
        # gentler alternative.
        self.instance_levels = [0, 1, 2]
        super().__init__(parser, "Setting Dataset parameters")
