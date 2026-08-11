"""A Potts prior on the occupancy field: commit every cell, minimise interface.

Two terms that look opposed and are not. Binarisation pushes each cell's opacity
to 0 or 1; the interface term pulls neighbours together. Together they are the
Potts / Mumford-Shah prior -- piecewise-constant occupancy with minimal boundary
area -- which is the classical statement of "make surfaces sharp".

The reason to want it here is that occupancy is currently a continuous mush.
`min(sigma_u, sigma_v)` separates within- from across-object Delaunay edges at
AUC 0.917, yet adding it to the multicut costs 3.3 mIoU, because it is not
conditionally independent of feature distance. A binary field turns it into an
actual "do these two cells touch" test, which is the structural signal the
Delaunay graph is supposed to provide and currently does not.

The second, less obvious prediction: a hollow shell has two interfaces, a solid
object has one, so minimising interface area should FILL interiors. If that
happens it addresses shells-over-vacuum -- 15.3% of cells sit at density 0.000
at every percentile -- as a consequence of the same loss rather than a separate
mechanism. See docs/specs/2026-08-11-occupancy-potts-prior-design.md.
"""

import torch

from radfoam_model.instance_graph import undirected_edges


class CellGeometry:
    """Neighbour distances, cached and invalidated when the mesh changes.

    Both terms need the Delaunay neighbour distances -- the interface weight is
    a face-area proxy built from them, and the opacity conversion needs a cell
    extent. Rebuilding costs a pass over ~15M edges, so it is cached.

    Keying the cache on point count alone is WRONG and was: between the end of
    densification and freeze_points the sites still move and the triangulation
    is rebuilt periodically, so connectivity changes while the count does not.
    The TV term then penalises differences between cells that are no longer
    neighbours, silently -- the indices stay in range, so nothing crashes. The
    edge count catches a rebuild, and the periodic refresh catches sites that
    drifted without changing it.
    """

    def __init__(self, refresh_every=100):
        self.n_points = None
        self.n_directed = None
        self.age = 0
        self.refresh_every = refresh_every
        self.edges = None
        self.face_weight = None
        self.extent = None

    def refresh(self, points, adjacency, offsets):
        unchanged = (
            self.n_points == points.shape[0]
            and self.n_directed == adjacency.shape[0]
            and self.age < self.refresh_every
        )
        if unchanged:
            self.age += 1
            return
        with torch.no_grad():
            edges = undirected_edges(adjacency, offsets)
            separation = (points[edges[:, 0]] - points[edges[:, 1]]).norm(dim=-1)
            # Voronoi face areas are not exposed by the tracer. In a locally
            # uniform tessellation faces scale as spacing^2 while separations
            # scale as spacing, so separation^2 is dimensionally right and
            # adapts to local point density for free.
            self.face_weight = separation.square()
            # Characteristic extent of a cell: half the mean distance to its
            # neighbours. Turns an unbounded density into a bounded opacity.
            total = torch.zeros(points.shape[0], device=points.device)
            count = torch.zeros_like(total)
            for column in (0, 1):
                total.index_add_(0, edges[:, column], separation)
                count.index_add_(0, edges[:, column],
                                 torch.ones_like(separation))
            self.extent = 0.5 * total / count.clamp(min=1)
            self.edges = edges
            self.n_points = points.shape[0]
            self.n_directed = adjacency.shape[0]
            self.age = 0
        return self


def cell_opacity(density, extent):
    """Bounded, scale-free stand-in for "is this cell solid".

    sigma is activation_scale * softplus(raw) and unbounded, so "push toward 1"
    is not defined on it. Opacity over the cell's own extent is, and it stays
    comparable between a dense foreground cell and a huge background one.
    """
    return -torch.expm1(-density.reshape(-1) * extent)


def occupancy_loss(density, geometry, penalty="entropy", sample=None,
                   generator=None):
    """(binarisation, interface) as two scalars, un-weighted.

    Returned separately so both can be logged and weighted independently -- the
    interaction between them is the whole question, and a single summed number
    would hide which one is moving.
    """
    alpha = cell_opacity(density, geometry.extent).clamp(1e-6, 1 - 1e-6)

    if penalty == "entropy":
        # Bounded in [0, log 2]. Reads as: every cell must commit. Its gradient
        # log((1-a)/a) diverges at the extremes, which is the runaway risk the
        # quadratic fallback exists for.
        binarisation = -(alpha * alpha.log()
                         + (1 - alpha) * (1 - alpha).log()).mean()
    elif penalty == "quadratic":
        binarisation = (alpha * (1 - alpha)).mean()
    else:
        raise ValueError(f"unknown penalty {penalty!r}")

    edges, weight = geometry.edges, geometry.face_weight
    if sample is not None and sample < edges.shape[0]:
        pick = torch.randint(edges.shape[0], (sample,), device=edges.device,
                             generator=generator)
        edges, weight = edges[pick], weight[pick]
    # L1, not L2. L1 is the total-variation / minimal-surface prior and permits
    # sharp jumps; L2 would smooth exactly the boundary this exists to sharpen.
    jump = (alpha[edges[:, 0]] - alpha[edges[:, 1]]).abs()
    interface = (weight * jump).sum() / weight.sum().clamp(min=1e-12)
    return binarisation, interface


@torch.no_grad()
def occupancy_report(density, geometry):
    """Undecided fraction and the opacity histogram -- gate 2 of the probe."""
    alpha = cell_opacity(density, geometry.extent)
    undecided = ((alpha > 0.1) & (alpha < 0.9)).float().mean()
    return {
        "undecided": undecided.item(),
        "solid": (alpha >= 0.9).float().mean().item(),
        "empty": (alpha <= 0.1).float().mean().item(),
        "mean": alpha.mean().item(),
    }
