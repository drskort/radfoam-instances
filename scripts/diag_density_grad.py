"""Diagnose the density-gradient discrepancy.

Two controls:

1. Sweep the finite-difference step. Central differences carry O(eps^2) error,
   so if the mismatch is truncation it shrinks quadratically as eps shrinks --
   down to the float32 round-off floor, which grows as 1/eps. A real bug does
   not move.
2. Run the identical check against the PHOTOMETRIC density gradient, which is
   upstream code and independent of the feature work. If that mismatches in the
   same places by the same amount, the test is at fault, not the kernel.
"""

import torch

import radfoam

FEAT_DIM, SH_DEGREE, NUM_POINTS, NUM_RAYS = 16, 0, 2048, 32
SH_DIM = 3 * (1 + SH_DEGREE) ** 2


def setup():
    torch.manual_seed(0)
    pts = torch.rand(NUM_POINTS, 3, device="cuda")
    tri = radfoam.Triangulation(pts)
    pts = pts[tri.permutation().to(torch.long)].contiguous()
    adj, off = tri.point_adjacency(), tri.point_adjacency_offsets()

    origin = torch.tensor([[0.5, 0.5, -2.0]], device="cuda").expand(NUM_RAYS, 3)
    direction = torch.nn.functional.normalize(
        torch.randn(NUM_RAYS, 3, device="cuda") * 0.15
        + torch.tensor([0.0, 0.0, 1.0], device="cuda"), dim=-1)
    rays = torch.cat([origin.contiguous(), direction.contiguous()], dim=-1)
    start = torch.zeros(NUM_RAYS, dtype=torch.uint32, device="cuda")

    attrs = torch.zeros(NUM_POINTS, 1 + SH_DIM + FEAT_DIM, device="cuda")
    attrs[:, :SH_DIM] = 0.5
    attrs[:, SH_DIM] = 3.0
    attrs[:, SH_DIM + 1:] = 0.1 * torch.randn(NUM_POINTS, FEAT_DIM, device="cuda")
    return pts, adj, off, rays, start, attrs


def main():
    pipe = radfoam.create_pipeline(SH_DEGREE, FEAT_DIM, torch.float32)
    pts, adj, off, rays, start, attrs = setup()

    def render(a):
        return pipe.trace_forward(pts, a, adj, off, rays, start,
                                  return_contribution=True)

    out = render(attrs)
    contribution = out["contribution"].squeeze(-1)
    hit = torch.nonzero(contribution > 1e-4).squeeze(-1)
    order = torch.argsort(contribution[hit], descending=True)
    sample = hit[order[:6]].tolist()

    g_feat = torch.randn(NUM_RAYS, FEAT_DIM, device="cuda")
    g_rgba = torch.randn(NUM_RAYS, 4, device="cuda")

    def analytic_density(feature_loss):
        """attr_grad from either the feature loss or the photometric loss."""
        grads = pipe.trace_backward(
            pts, attrs, adj, off, rays, start,
            out["rgba"],
            torch.zeros_like(out["rgba"]) if feature_loss else g_rgba,
            ray_feature=out["feature"] if feature_loss else None,
            ray_feature_grad=g_feat if feature_loss else None,
            instance_guided_geometry=feature_loss,
        )
        return grads["attr_grad"][:, SH_DIM]

    def loss_of(a, feature_loss):
        o = render(a)
        return ((o["feature"] * g_feat).sum() if feature_loss
                else (o["rgba"] * g_rgba).sum())

    for feature_loss in (True, False):
        label = "FEATURE loss -> density" if feature_loss else "RGB loss -> density (control)"
        print(f"\n=== {label} ===")
        a_grad = analytic_density(feature_loss)
        print(f"{'point':>7} " + "".join(f"{'eps=' + f'{e:g}':>14}"
                                         for e in (1e-2, 5e-3, 2e-3, 1e-3))
              + f"{'analytic':>13}")
        for idx in sample:
            row = f"{idx:7d} "
            for eps in (1e-2, 5e-3, 2e-3, 1e-3):
                plus, minus = attrs.clone(), attrs.clone()
                plus[idx, SH_DIM] += eps
                minus[idx, SH_DIM] -= eps
                num = (loss_of(plus, feature_loss)
                       - loss_of(minus, feature_loss)) / (2 * eps)
                row += f"{num.item():14.6f}"
            row += f"{a_grad[idx].item():13.6f}"
            print(row)


if __name__ == "__main__":
    main()
