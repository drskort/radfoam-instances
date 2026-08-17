"""Finite-difference check of the variance-loss gradient.

Companion to gradcheck_features.py. That one injects a fixed dL/dF and checks a
single composited output; this one exercises the second output V = sum w f^2 and
the coupling between them, s^2 = V - F^2.

The coupling is the part worth testing. dL/df_n = w_n(g_F - 2F g_V) + 2 w_n f_n g_V
has two halves that are individually plausible and cancel to something wrong if
either sign is off -- and OpenSplat3D's own implementation drops the second one,
so "looks like the reference" is not evidence of correctness here.

    srun -p a40-lo --gres=gpu:1 --time=00:20:00 \
        .venv/bin/python scripts/gradcheck_variance.py
"""

import torch

import radfoam

FEAT_DIM = 16
SH_DEGREE = 0
NUM_POINTS = 2048
NUM_RAYS = 32
SH_DIM = 3 * (1 + SH_DEGREE) ** 2
EPS = 1e-2
ATOL = 2e-5
RTOL = 5e-3
DENSITY_EPS = 1e-3
DENSITY_ATOL = 3e-4


def build_scene(seed=0):
    torch.manual_seed(seed)
    points = torch.rand(NUM_POINTS, 3, device="cuda")
    tri = radfoam.Triangulation(points)
    perm = tri.permutation().to(torch.long)
    points = points[perm].contiguous()
    return points, tri.point_adjacency(), tri.point_adjacency_offsets()


def build_rays():
    origin = torch.tensor([[0.5, 0.5, -2.0]], device="cuda").expand(NUM_RAYS, 3)
    direction = torch.nn.functional.normalize(
        torch.randn(NUM_RAYS, 3, device="cuda") * 0.15
        + torch.tensor([0.0, 0.0, 1.0], device="cuda"),
        dim=-1,
    )
    return torch.cat([origin.contiguous(), direction.contiguous()], dim=-1)


def variance_loss(out):
    """The trained objective: mean over rays and channels of (V - F^2)^2."""
    F = out["feature"].float()
    V = out["feature_squared"].float()
    return ((V - F.square()) ** 2).mean()


def upstream(out):
    """dL/dF and dL/dV for that loss, computed in torch so the kernel is the
    only thing under test."""
    F = out["feature"].float().detach().requires_grad_(True)
    V = out["feature_squared"].float().detach().requires_grad_(True)
    ((V - F.square()) ** 2).mean().backward()
    return F.grad.contiguous(), V.grad.contiguous()


def check(guided_geometry):
    pipe = radfoam.create_pipeline(SH_DEGREE, FEAT_DIM, torch.float32)
    points, adjacency, offsets = build_scene()
    rays = build_rays()
    start = torch.zeros(NUM_RAYS, dtype=torch.uint32, device="cuda")

    attrs = torch.zeros(NUM_POINTS, 1 + SH_DIM + FEAT_DIM, device="cuda")
    attrs[:, :SH_DIM] = 0.5
    attrs[:, SH_DIM] = 3.0
    # Unit-scale features, deliberately. At 0.1 the variance is ~1e-4, the
    # loss ~1e-8 and every gradient sits under the 2e-5 floor that float32
    # central differences can resolve -- the test then measures round-off
    # rather than the kernel. A trained checkpoint carries feature std ~0.5
    # with s^2 up to 22, so unit scale is also the honest regime.
    attrs[:, SH_DIM + 1:] = torch.randn(NUM_POINTS, FEAT_DIM, device="cuda")

    def render(a):
        return pipe.trace_forward(
            points, a, adjacency, offsets, rays, start, return_contribution=True
        )

    out = render(attrs)
    contribution = out["contribution"].squeeze(-1)
    g_F, g_V = upstream(out)

    grads = pipe.trace_backward(
        points, attrs, adjacency, offsets, rays, start,
        out["rgba"],
        torch.zeros_like(out["rgba"]),          # no photometric gradient
        ray_feature=out["feature"],
        ray_feature_grad=g_F,
        ray_feature_squared=out["feature_squared"],
        ray_feature_squared_grad=g_V,
        instance_guided_geometry=guided_geometry,
    )
    analytic = grads["attr_grad"]

    hit = torch.nonzero(contribution > 1e-4).squeeze(-1)
    print(f"loss {variance_loss(out).item():.6f} | "
          f"primitives hit: {hit.numel()} of {NUM_POINTS}")
    if hit.numel() == 0:
        raise SystemExit("no primitives hit -- adjust the ray setup")
    order = torch.argsort(contribution[hit], descending=True)
    sample = hit[order[:8]]

    # ---- feature channels -------------------------------------------------
    worst, failures, tested = 0.0, 0, 0
    for point_idx in sample.tolist():
        for col in range(SH_DIM + 1, SH_DIM + 1 + FEAT_DIM, 5):
            plus, minus = attrs.clone(), attrs.clone()
            plus[point_idx, col] += EPS
            minus[point_idx, col] -= EPS
            numeric = (
                variance_loss(render(plus)) - variance_loss(render(minus))
            ) / (2 * EPS)
            a = analytic[point_idx, col]
            err = abs(numeric.item() - a.item())
            worst = max(worst, err)
            tested += 1
            if err > ATOL + RTOL * abs(a.item()):
                failures += 1
                if failures <= 3:
                    print(f"    point {point_idx} col {col}: "
                          f"analytic {a.item():+.6f} numeric {numeric.item():+.6f}")
    print(f"  feature: worst |err| {worst:.2e} over {tested} entries "
          f"(tol {ATOL:.1e} + {RTOL:.1e}*|grad|), failing: {failures}")

    # ---- density ----------------------------------------------------------
    density_worst, density_max, density_tested = 0.0, 0.0, 0
    for point_idx in sample[:6].tolist():
        col = SH_DIM
        plus, minus = attrs.clone(), attrs.clone()
        plus[point_idx, col] += DENSITY_EPS
        minus[point_idx, col] -= DENSITY_EPS
        numeric = (
            variance_loss(render(plus)) - variance_loss(render(minus))
        ) / (2 * DENSITY_EPS)
        a = analytic[point_idx, col]
        density_worst = max(density_worst, abs(numeric.item() - a.item()))
        density_max = max(density_max, abs(a.item()))
        density_tested += 1
    ok = density_worst <= DENSITY_ATOL + RTOL * max(density_max, 1e-12)
    if guided_geometry:
        print(f"  density: worst |err| {density_worst:.2e}, "
              f"max |grad| {density_max:.2e} over {density_tested} points -> "
              f"{'PASS' if ok else 'FAIL'}")
    else:
        # With the coupling off the variance must not touch geometry at all.
        print(f"  density: max |grad| {density_max:.2e} "
              f"(must be ~0 with instance_guided_geometry off) -> "
              f"{'PASS' if density_max < 1e-9 else 'FAIL'}")
        ok = density_max < 1e-9
    return failures == 0 and ok


if __name__ == "__main__":
    all_ok = True
    for guided in (False, True):
        print(f"\ninstance_guided_geometry = {guided}")
        all_ok &= check(guided)
    print("\n" + ("ALL PASS" if all_ok else "FAILURES -- see above"))
