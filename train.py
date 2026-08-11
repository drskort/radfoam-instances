import os
import uuid
import yaml
import gc
import time
import numpy as np
from PIL import Image
import configargparse
import tqdm
import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data_loader import DataHandler
from configs import *
from radfoam_model.instance_loss import multi_level_instance_loss
from radfoam_model.occupancy_loss import (
    CellGeometry,
    occupancy_loss,
    occupancy_report,
)
from radfoam_model.scene import RadFoamScene
from radfoam_model.utils import psnr
import radfoam


seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


def train(args, pipeline_args, model_args, optimizer_args, dataset_args):
    device = torch.device(model_args.device)
    # Setting up output directory
    if not pipeline_args.debug:
        if len(pipeline_args.experiment_name) == 0:
            unique_str = str(uuid.uuid4())[:8]
            experiment_name = f"{dataset_args.scene}@{unique_str}"
        else:
            experiment_name = pipeline_args.experiment_name
        out_dir = f"output/{experiment_name}"
        writer = SummaryWriter(out_dir, purge_step=0)
        os.makedirs(f"{out_dir}/test", exist_ok=True)

        def represent_list_inline(dumper, data):
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )

        yaml.add_representer(list, represent_list_inline)

        # Save the arguments to a YAML file
        with open(f"{out_dir}/config.yaml", "w") as yaml_file:
            yaml.dump(vars(args), yaml_file, default_flow_style=False)

    # Setting up dataset
    iter2downsample = dict(
        zip(
            dataset_args.downsample_iterations,
            dataset_args.downsample,
        )
    )
    train_data_handler = DataHandler(
        dataset_args, rays_per_batch=1_000_000, device=device
    )
    downsample = iter2downsample[0]
    train_data_handler.reload(split="train", downsample=downsample)

    test_data_handler = DataHandler(
        dataset_args, rays_per_batch=0, device=device
    )
    test_data_handler.reload(
        split="test", downsample=min(dataset_args.downsample)
    )
    test_ray_batch_fetcher = radfoam.BatchFetcher(
        test_data_handler.rays, batch_size=1, shuffle=False
    )
    test_rgb_batch_fetcher = radfoam.BatchFetcher(
        test_data_handler.rgbs, batch_size=1, shuffle=False
    )

    # Define viewer settings
    viewer_options = {
        "camera_pos": train_data_handler.viewer_pos,
        "camera_up": train_data_handler.viewer_up,
        "camera_forward": train_data_handler.viewer_forward,
    }

    # Setting up pipeline
    rgb_loss = nn.SmoothL1Loss(reduction="none")

    # Setting up model
    model = RadFoamScene(
        args=model_args,
        device=device,
        points=train_data_handler.points3D,
        points_colors=train_data_handler.points3D_colors,
    )

    # Fine-tune an existing scene instead of building one. Only useful past
    # freeze_points, where the triangulation is already fixed: densification
    # and point motion are both over, so a short continuation moves densities
    # and attributes and nothing else. That is what isolates a density prior
    # from geometry adapting around it.
    if pipeline_args.resume_from:
        model.load_pt(pipeline_args.resume_from)
        print(f"resumed from {pipeline_args.resume_from}: "
              f"{model.primal_points.shape[0]} points", flush=True)
        if pipeline_args.densify_until > 0:
            raise SystemExit(
                "resume_from with densification still enabled would rebuild "
                "the triangulation and destroy the checkpoint being probed; "
                "pass --densify_from 0 --densify_until 0 --freeze_points 0."
            )

    # Setting up optimizer
    model.declare_optimizer(
        args=optimizer_args,
        warmup=pipeline_args.densify_from,
        max_iterations=pipeline_args.iterations,
    )

    def test_render(
        test_data_handler, ray_batch_fetcher, rgb_batch_fetcher, debug=False
    ):
        rays = test_data_handler.rays
        points, _, _, _ = model.get_trace_data()
        start_points = model.get_starting_point(
            rays[:, 0, 0].cuda(), points, model.aabb_tree
        )

        psnr_list = []
        with torch.no_grad():
            for i in range(rays.shape[0]):
                ray_batch = ray_batch_fetcher.next()[0]
                rgb_batch = rgb_batch_fetcher.next()[0]
                output, *_ = model(ray_batch, start_points[i])

                # White background
                opacity = output[..., -1:]
                rgb_output = output[..., :3] + (1 - opacity)
                rgb_output = rgb_output.reshape(*rgb_batch.shape).clip(0, 1)

                img_psnr = psnr(rgb_output, rgb_batch).mean()
                psnr_list.append(img_psnr)
                torch.cuda.synchronize()

                if not debug:
                    error = np.uint8((rgb_output - rgb_batch).cpu().abs() * 255)
                    rgb_output = np.uint8(rgb_output.cpu() * 255)
                    rgb_batch = np.uint8(rgb_batch.cpu() * 255)

                    im = Image.fromarray(
                        np.concatenate([rgb_output, rgb_batch, error], axis=1)
                    )
                    im.save(
                        f"{out_dir}/test/rgb_{i:03d}_psnr_{img_psnr:.3f}.png"
                    )

        average_psnr = sum(psnr_list) / len(psnr_list)
        if not debug:
            f = open(f"{out_dir}/metrics.txt", "w")
            f.write(f"Average PSNR: {average_psnr}")
            f.close()

        return average_psnr

    cell_geometry = CellGeometry()

    def train_loop(viewer):
        print("Training")

        torch.cuda.synchronize()

        # The model attribute is toggled per iteration below, so remember what
        # the run actually asked for.
        guided_geometry = model.instance_guided_geometry

        data_iterator = train_data_handler.get_iter()
        ray_batch, rgb_batch, alpha_batch, label_batch = next(data_iterator)

        triangulation_update_period = 1
        iters_since_update = 1
        iters_since_densification = 0
        next_densification_after = 1

        with tqdm.trange(
            pipeline_args.start_iteration, pipeline_args.iterations
        ) as train:
            for i in train:
                if viewer is not None:
                    model.update_viewer(viewer)
                    viewer.step(i)

                if i in iter2downsample and i:
                    downsample = iter2downsample[i]
                    train_data_handler.reload(
                        split="train", downsample=downsample
                    )
                    data_iterator = train_data_handler.get_iter()
                    ray_batch, rgb_batch, alpha_batch, label_batch = next(
                    data_iterator
                )

                # Features start as noise, so coupling them to geometry from
                # iteration 0 pushes points around for no reason -- on
                # figurines that drove the cells into a degenerate
                # configuration and the incremental Delaunay rebuild died with
                # an illegal memory access at iteration 195. Let the features
                # become meaningful first, then let them move geometry.
                if model.feat_dim > 0:
                    model.instance_guided_geometry = (
                        guided_geometry
                        and i >= pipeline_args.instance_geometry_from
                    )

                depth_quantiles = (
                    torch.rand(*ray_batch.shape[:-1], 2, device=device)
                    .sort(dim=-1, descending=True)
                    .values
                )

                rgba_output, _feature, _feature_squared, depth, _, _, _ = model(
                    ray_batch,
                    depth_quantiles=depth_quantiles,
                )

                # White background
                opacity = rgba_output[..., -1:]
                if pipeline_args.white_background:
                    rgb_output = rgba_output[..., :3] + (1 - opacity)
                else:
                    rgb_output = rgba_output[..., :3]

                color_loss = rgb_loss(rgb_batch, rgb_output)
                opacity_loss = ((alpha_batch - opacity) ** 2).mean()

                valid_depth_mask = (depth > 0).all(dim=-1)
                quant_loss = (depth[..., 0] - depth[..., 1]).abs()
                quant_loss = (quant_loss * valid_depth_mask).mean()
                w_depth = pipeline_args.quantile_weight * min(
                    2 * i / pipeline_args.iterations, 1
                )

                loss = color_loss.mean() + opacity_loss + w_depth * quant_loss

                # Contrastive instance features, supervised per view from the
                # precomputed SAM label maps. Skipped when no masks are present
                # or the model carries no feature channels.
                instance_stats = None
                variance_loss = None
                feature_variance = None
                if (
                    label_batch is not None
                    and model.feat_dim > 0
                    and pipeline_args.instance_weight > 0
                ):
                    instance_stats = multi_level_instance_loss(
                        _feature,
                        label_batch.to(torch.long),
                        gamma=pipeline_args.instance_gamma,
                        weights=(
                            pipeline_args.instance_pos_weight,
                            pipeline_args.instance_neg_weight,
                        ),
                        level_weights=pipeline_args.instance_level_weights,
                    )
                    loss = loss + (
                        pipeline_args.instance_weight * instance_stats["total"]
                    )

                # Variance loss. Unsupervised -- it asks each ray to hit cells
                # that agree with each other and needs no masks, so it is not
                # gated on the instance loss.
                #
                # NOTE the subtraction below is deliberately NOT under
                # no_grad: autograd carries the d(s^2)/dF = -2F path into
                # grad_feature, and the kernel relies on that. See
                # docs/variance_backward.md.
                if (
                    _feature_squared is not None
                    and model.feat_dim > 0
                    and pipeline_args.variance_weight > 0
                ):
                    feature_variance = _feature_squared - torch.square(_feature)
                    variance_loss = feature_variance.pow(2).mean()
                    loss = loss + (pipeline_args.variance_weight * variance_loss)

                # Potts prior on occupancy. Geometry-only and unsupervised, so
                # it is gated on neither masks nor features.
                if (pipeline_args.occupancy_bin_weight > 0
                        or pipeline_args.occupancy_tv_weight > 0) and (
                        i >= pipeline_args.occupancy_from):
                    points, _, adjacency, offsets = model.get_trace_data()
                    cell_geometry.refresh(points, adjacency, offsets)
                    binarisation, interface = occupancy_loss(
                        model.get_primal_density(), cell_geometry,
                        penalty=pipeline_args.occupancy_penalty,
                        sample=(pipeline_args.occupancy_edge_sample or None),
                    )
                    loss = loss + (
                        pipeline_args.occupancy_bin_weight * binarisation
                        + pipeline_args.occupancy_tv_weight * interface
                    )


                model.optimizer.zero_grad(set_to_none=True)

                # Hide latency of data loading behind the backward pass
                event = torch.cuda.Event()
                event.record()
                loss.backward()
                event.synchronize()
                ray_batch, rgb_batch, alpha_batch, label_batch = next(
                    data_iterator
                )

                model.optimizer.step()
                model.update_learning_rate(i)

                train.set_postfix(color_loss=f"{color_loss.mean().item():.5f}")

                if i % 100 == 99 and not pipeline_args.debug:
                    if cell_geometry.n_points is not None:
                        stats = occupancy_report(
                            model.get_primal_density(), cell_geometry)
                        for key, value in stats.items():
                            writer.add_scalar(f"occupancy/{key}", value, i)
                        writer.add_scalar("loss/occupancy_bin",
                                          binarisation.item(), i)
                        writer.add_scalar("loss/occupancy_tv",
                                          interface.item(), i)
                    writer.add_scalar("train/rgb_loss", color_loss.mean(), i)
                    # Is the instance loss actually moving the features? A dead
                    # gradient here means the loss is not reaching att_feat, and
                    # shows up in minutes instead of after a full render.
                    if instance_stats is not None:
                        writer.add_scalar(
                            "instance/total", instance_stats["total"], i
                        )
                        for key, value in instance_stats.items():
                            if key.startswith("l") and "_" in key:
                                writer.add_scalar(f"instance/{key}", value, i)
                        if model.att_feat.grad is not None:
                            writer.add_scalar(
                                "instance/att_feat_grad_norm",
                                model.att_feat.grad.norm(),
                                i,
                            )
                        writer.add_scalar(
                            "instance/att_feat_std", model.att_feat.std(), i
                        )
                    # One group with every term on the same axes, weighted as
                    # it actually enters the total -- the raw values differ by
                    # orders of magnitude and are not comparable.
                    writer.add_scalar("loss/total", loss, i)
                    writer.add_scalar("loss/rgb", color_loss.mean(), i)
                    writer.add_scalar("loss/opacity", opacity_loss, i)
                    if instance_stats is not None:
                        writer.add_scalar(
                            "loss/instance_weighted",
                            pipeline_args.instance_weight
                            * instance_stats["total"],
                            i,
                        )
                        writer.add_scalar(
                            "loss/instance_raw", instance_stats["total"], i
                        )
                    if variance_loss is not None:
                        writer.add_scalar(
                            "loss/variance_weighted",
                            pipeline_args.variance_weight * variance_loss,
                            i,
                        )
                        writer.add_scalar("loss/variance_raw", variance_loss, i)
                        writer.add_scalar("variance/loss", variance_loss, i)
                        with torch.no_grad():
                            writer.add_scalar(
                                "variance/s2_mean", feature_variance.mean(), i
                            )
                            writer.add_scalar(
                                "variance/s2_max", feature_variance.max(), i
                            )
                            # Should never go meaningfully below zero; a
                            # negative here means the two accumulators have
                            # drifted apart in the kernel.
                            writer.add_scalar(
                                "variance/s2_min", feature_variance.min(), i
                            )
                    num_points = model.primal_points.shape[0]
                    writer.add_scalar("test/num_points", num_points, i)

                    test_psnr = test_render(
                        test_data_handler,
                        test_ray_batch_fetcher,
                        test_rgb_batch_fetcher,
                        True,
                    )
                    writer.add_scalar("test/psnr", test_psnr, i)

                    writer.add_scalar(
                        "lr/points_lr", model.xyz_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/density_lr", model.den_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/attr_lr", model.attr_dc_scheduler_args(i), i
                    )

                # A resumed run must not touch geometry: the point of the
                # probe is that the triangulation is held fixed while densities
                # move, and prune_and_densify would rebuild the very
                # checkpoint being measured. densify_from = 0 does not express
                # this -- it makes the counter run every step -- so the guard
                # is explicit.
                # A resumed run must not densify -- prune_and_densify would
                # rebuild the very checkpoint being continued. It MAY still
                # retriangulate, and must, whenever the sites are free to
                # move: holding a stale mesh under moving points is exactly
                # the staleness CellGeometry was fixed for. Frozen sites
                # (points_lr 0, the fine-tune probe) need neither.
                frozen = optimizer_args.points_lr_init == 0
                if pipeline_args.resume_from and frozen:
                    iters_since_update = 0
                elif iters_since_update >= triangulation_update_period:
                    if pipeline_args.debug_triangulation:
                        print(f"[{i}] retriangulating...", flush=True)
                        _t0 = time.time()
                    model.update_triangulation(incremental=True)
                    if pipeline_args.debug_triangulation:
                        print(f"[{i}] retriangulated in "
                              f"{time.time() - _t0:.2f}s", flush=True)
                    iters_since_update = 0

                    if triangulation_update_period < 100:
                        triangulation_update_period += 2

                iters_since_update += 1
                if i + 1 >= pipeline_args.densify_from:
                    iters_since_densification += 1

                if (
                    not pipeline_args.resume_from
                    and iters_since_densification == next_densification_after
                    and model.primal_points.shape[0]
                    < 0.9 * model.num_final_points
                ):
                    point_error, point_contribution = model.collect_error_map(
                        train_data_handler, pipeline_args.white_background
                    )
                    model.prune_and_densify(
                        point_error,
                        point_contribution,
                        pipeline_args.densify_factor,
                    )

                    model.update_triangulation(incremental=False)
                    triangulation_update_period = 1
                    gc.collect()

                    # Linear growth
                    iters_since_densification = 0
                    next_densification_after = int(
                        (
                            (pipeline_args.densify_factor - 1)
                            * model.primal_points.shape[0]
                            * (
                                pipeline_args.densify_until
                                - pipeline_args.densify_from
                            )
                        )
                        / (model.num_final_points - model.num_init_points)
                    )
                    next_densification_after = max(
                        next_densification_after, 100
                    )

                if i == optimizer_args.freeze_points:
                    model.update_triangulation(incremental=False)

                if viewer is not None and viewer.is_closed():
                    break

                # Periodic checkpoint. A multi-hour run that only saves at the
                # end is a single point of failure -- a walltime overrun loses
                # everything. Also lets intermediate scenes be rendered while
                # training continues.
                if (
                    pipeline_args.checkpoint_every > 0
                    and (i + 1) % pipeline_args.checkpoint_every == 0
                ):
                    model.save_pt(f"{out_dir}/model.pt")
                    model.save_pt(f"{out_dir}/model_{i + 1:06d}.pt")
                    train.set_postfix_str(
                        f"color_loss={color_loss.mean().item():.5f} (saved)"
                    )

        model.save_ply(f"{out_dir}/scene.ply")
        model.save_pt(f"{out_dir}/model.pt")
        del data_iterator

    if pipeline_args.viewer:
        model.show(
            train_loop, iterations=pipeline_args.iterations, **viewer_options
        )
    else:
        train_loop(viewer=None)
    if not pipeline_args.debug:
        writer.close()

    test_render(
        test_data_handler,
        test_ray_batch_fetcher,
        test_rgb_batch_fetcher,
        pipeline_args.debug,
    )


def main():
    parser = configargparse.ArgParser(
        default_config_files=["arguments/mipnerf360_outdoor_config.yaml"]
    )

    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Parse arguments
    args = parser.parse_args()

    train(
        args,
        pipeline_params.extract(args),
        model_params.extract(args),
        optimization_params.extract(args),
        dataset_params.extract(args),
    )


if __name__ == "__main__":
    main()
