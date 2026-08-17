"""Finite-difference check of the instance-feature gradient.

A hand-written backward pass that is subtly wrong still trains, and produces
plausible-looking features. This is the only cheap way to know it is right.

Run on a GPU node:
    srun -p a40-lo --gres=gpu:1 --time=00:20:00 \
        .venv/bin/python scripts/gradcheck_features.py
"""

import torch

import radfoam

FEAT_DIM = 16
SH_DEGREE = 0
NUM_POINTS = 2048
NUM_RAYS = 32
SH_DIM = 3 * (1 + SH_DEGREE) ** 2
# The pipeline is float32, so the usual 1e-6 finite-difference step drowns in
# round-off. 1e-2 is large enough to survive it and small enough to stay linear.
EPS = 1e-2
# Finite differencing in float32 divides round-off by 2*EPS, so gradients below
# roughly 1e-5 cannot be measured this way at all -- hence an absolute floor
# alongside the relative tolerance.
ATOL = 2e-5
RTOL = 5e-3
# The density path is NONLINEAR in sigma, so central differences carry O(eps^2)
# truncation error that dominates on cells where alpha saturates. Features are
# linear, so EPS is exact for them; density needs a smaller step and a looser
# tolerance. Verified against the upstream photometric density gradient, which
# is equally noisy at the same primitives -- see scripts/diag_density_grad.py.
DENSITY_EPS = 1e-3
DENSITY_ATOL = 3e-4


def build_scene(seed=0):
    torch.manual_seed(seed)
    points = torch.rand(NUM_POINTS, 3, device="cuda")
    tri = radfoam.Triangulation(points)
    perm = tri.permutation().to(torch.long)
    points = points[perm].contiguous()
    return (
        points,
        tri.point_adjacency(),
        tri.point_adjacency_offsets(),
    )


def build_rays():
    origin = torch.tensor([[0.5, 0.5, -2.0]], device="cuda").expand(NUM_RAYS, 3)
    direction = torch.nn.functional.normalize(
        torch.randn(NUM_RAYS, 3, device="cuda") * 0.15
        + torch.tensor([0.0, 0.0, 1.0], device="cuda"),
        dim=-1,
    )
    return torch.cat([origin.contiguous(), direction.contiguous()], dim=-1)


def check(guided_geometry):
    """Run the check with the instance->geometry term on or off."""
    pipe = radfoam.create_pipeline(SH_DEGREE, FEAT_DIM, torch.float32)
    points, adjacency, offsets = build_scene()
    rays = build_rays()
    start = torch.zeros(NUM_RAYS, dtype=torch.uint32, device="cuda")

    attrs = torch.zeros(NUM_POINTS, 1 + SH_DIM + FEAT_DIM, device="cuda")
    attrs[:, :SH_DIM] = 0.5
    attrs[:, SH_DIM] = 3.0                                   # density
    attrs[:, SH_DIM + 1:] = 0.1 * torch.randn(NUM_POINTS, FEAT_DIM, device="cuda")

    # A fixed random dL/dfeature, so the loss is a plain linear functional of
    # the rendered feature map and its gradient is exactly what we injected.
    g = torch.randn(NUM_RAYS, FEAT_DIM, device="cuda")

    def render(a):
        return pipe.trace_forward(
            points, a, adjacency, offsets, rays, start, return_contribution=True
        )

    out = render(attrs)
    contribution = out["contribution"].squeeze(-1)
    loss = (out["feature"] * g).sum()

    grads = pipe.trace_backward(
        points,
        attrs,
        adjacency,
        offsets,
        rays,
        start,
        out["rgba"],
        torch.zeros_like(out["rgba"]),          # no photometric gradient
        ray_feature=out["feature"],
        ray_feature_grad=g,
        instance_guided_geometry=guided_geometry,
    )
    analytic = grads["attr_grad"]

    # Only primitives the rays actually reach can have a gradient; the rest
    # carry no information and would just report 0 == 0.
    hit = torch.nonzero(contribution > 1e-4).squeeze(-1)
    print(f"loss {loss.item():.4f} | primitives hit: {hit.numel()} of {NUM_POINTS}")
    if hit.numel() == 0:
        raise SystemExit("no primitives hit -- adjust the ray setup")

    # Test where the signal is strongest. A finite difference on a primitive
    # with negligible contribution measures round-off, not the gradient.
    order = torch.argsort(contribution[hit], descending=True)
    sample = hit[order[:8]]

    print(f"\n{'point':>7} {'ch':>3} {'numeric':>12} {'analytic':>12} "
          f"{'abs err':>10} {'rel':>9}")
    worst = 0.0
    failures = 0
    for point_idx in sample.tolist():
        for channel in (0, FEAT_DIM // 2, FEAT_DIM - 1):
            col = SH_DIM + 1 + channel

            plus = attrs.clone()
            plus[point_idx, col] += EPS
            minus = attrs.clone()
            minus[point_idx, col] -= EPS

            numeric = (
                (render(plus)["feature"] * g).sum()
                - (render(minus)["feature"] * g).sum()
            ) / (2 * EPS)
            a = analytic[point_idx, col]

            abs_err = abs(numeric.item() - a.item())
            rel = abs_err / max(abs(a.item()), 1e-12)
            # Mixed tolerance, as torch.autograd.gradcheck uses. A pure relative
            # test is meaningless where the gradient is near zero: the finite
            # difference divides float32 round-off by 2*EPS, which floors the
            # measurable gradient at ~1e-5 regardless of correctness.
            ok = abs_err <= ATOL + RTOL * abs(a.item())
            worst = max(worst, abs_err)
            print(f"{point_idx:7d} {channel:3d} {numeric.item():12.5f} "
                  f"{a.item():12.5f} {abs_err:10.2e} {rel:9.1e} "
                  f"{'ok' if ok else 'BAD'}")
            if not ok:
                failures += 1

    print(f"\nfeature gradient: worst absolute error {worst:.2e}  (tolerance "
          f"{ATOL:.1e} + {RTOL:.1e}*|grad|), failing entries: {failures}")

    # --- the density gradient, i.e. term (b) -----------------------------
    # A feature-only loss must leave density untouched when the geometry term
    # is off, and must match finite differences when it is on.
    print(f"\n--- density gradient (instance_guided_geometry="
          f"{guided_geometry}) ---")
    print(f"{'point':>7} {'numeric':>12} {'analytic':>12} {'abs err':>10}")
    density_worst = 0.0
    for point_idx in sample[:6].tolist():
        col = SH_DIM
        plus = attrs.clone()
        plus[point_idx, col] += DENSITY_EPS
        minus = attrs.clone()
        minus[point_idx, col] -= DENSITY_EPS
        numeric = (
            (render(plus)["feature"] * g).sum()
            - (render(minus)["feature"] * g).sum()
        ) / (2 * DENSITY_EPS)
        a = analytic[point_idx, col]
        abs_err = abs(numeric.item() - a.item())
        density_worst = max(density_worst, abs_err)
        print(f"{point_idx:7d} {numeric.item():12.5f} {a.item():12.5f} "
              f"{abs_err:10.2e}")

    max_density = analytic[hit, SH_DIM].abs().max().item()
    print(f"\nmax |analytic density gradient| = {max_density:.3e}")
    if guided_geometry:
        # Both that it is non-zero (the term fires) and that it is right.
        ok = (density_worst <= DENSITY_ATOL + RTOL * max_density
              and max_density > 1e-4)
        print(f"density gradient worst abs error {density_worst:.2e} -> "
              f"{'PASS' if ok else 'FAIL'}")
    else:
        ok = max_density == 0.0
        print(f"expected exactly 0 with the term disabled -> "
              f"{'PASS' if ok else 'FAIL'}")
        failures += 0 if ok else 1
    if guided_geometry and not ok:
        failures += 1
    # Density must be untouched by a feature-only loss unless the geometry term
    # is enabled -- a direct check that the two paths are separate.
    density_grad = analytic[hit, SH_DIM].abs().max().item()
    print(f"max |density gradient| (expect 0 without instance_guided_geometry): "
          f"{density_grad:.3e}")
    print("PASS" if failures == 0 else "FAIL")
    return failures == 0


if __name__ == "__main__":
    results = []
    for guided in (False, True):
        print("=" * 68)
        print(f"instance_guided_geometry = {guided}")
        print("=" * 68)
        results.append(check(guided))
    print("\nOVERALL:", "PASS" if all(results) else "FAIL")
