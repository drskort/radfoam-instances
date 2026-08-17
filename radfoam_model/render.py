import torch


class ErrorBox:
    def __init__(self):
        self.ray_error = None
        self.point_error = None


class TraceRays(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pipeline,
        _points,
        _attributes,
        _point_adjacency,
        _point_adjacency_offsets,
        rays,
        start_point,
        depth_quantiles,
        return_contribution,
        instance_guided_geometry,
    ):
        ctx.rays = rays
        # Whether the instance-feature gradient also drives density and
        # geometry. Off by default; see TraceSettings in pipeline.h.
        ctx.instance_guided_geometry = instance_guided_geometry
        ctx.start_point = start_point
        ctx.depth_quantiles = depth_quantiles
        ctx.pipeline = pipeline
        ctx.points = _points
        ctx.attributes = _attributes
        ctx.point_adjacency = _point_adjacency
        ctx.point_adjacency_offsets = _point_adjacency_offsets

        results = pipeline.trace_forward(
            _points,
            _attributes,
            _point_adjacency,
            _point_adjacency_offsets,
            rays,
            start_point,
            depth_quantiles=depth_quantiles,
            return_contribution=return_contribution,
        )
        ctx.rgba = results["rgba"]
        # Needed by the backward pass to reconstruct "everything behind this
        # primitive", exactly as rgba is.
        ctx.feature = results.get("feature", None)
        ctx.feature_squared = results.get("feature_squared", None)
        ctx.depth_indices = results.get("depth_indices", None)

        errbox = ErrorBox()
        ctx.errbox = errbox

        return (
            results["rgba"],
            results.get("feature", None),
            results.get("feature_squared", None),
            results.get("depth", None),
            results.get("contribution", None),
            results["num_intersections"],
            errbox,
        )

    @staticmethod
    def backward(
        ctx,
        grad_rgba,
        grad_feature,
        grad_feature_squared,
        grad_depth,
        grad_contribution,
        grad_num_intersections,
        errbox_grad,
    ):
        del grad_contribution
        del grad_num_intersections
        del errbox_grad

        rays = ctx.rays
        start_point = ctx.start_point
        pipeline = ctx.pipeline
        rgba = ctx.rgba
        _points = ctx.points
        _attributes = ctx.attributes
        _point_adjacency = ctx.point_adjacency
        _point_adjacency_offsets = ctx.point_adjacency_offsets
        depth_quantiles = ctx.depth_quantiles

        results = pipeline.trace_backward(
            _points,
            _attributes,
            _point_adjacency,
            _point_adjacency_offsets,
            rays,
            start_point,
            rgba,
            grad_rgba,
            depth_quantiles,
            ctx.depth_indices,
            grad_depth,
            ctx.errbox.ray_error,
            ray_feature=ctx.feature,
            ray_feature_grad=grad_feature,
            ray_feature_squared=ctx.feature_squared,
            ray_feature_squared_grad=grad_feature_squared,
            instance_guided_geometry=ctx.instance_guided_geometry,
        )
        points_grad = results["points_grad"]
        attr_grad = results["attr_grad"]
        ctx.errbox.point_error = results.get("point_error", None)

        points_grad[~points_grad.isfinite()] = 0
        attr_grad[~attr_grad.isfinite()] = 0

        del (
            ctx.rays,
            ctx.start_point,
            ctx.pipeline,
            ctx.rgba,
            ctx.points,
            ctx.attributes,
            ctx.point_adjacency,
            ctx.point_adjacency_offsets,
            ctx.depth_quantiles,
            ctx.feature,
        )
        return (
            None,  # pipeline
            points_grad,  # _points
            attr_grad,  # _attributes
            None,  # _point_adjacency
            None,  # _point_adjacency_offsets
            None,  # rays
            None,  # start_point
            None,  # depth_quantiles
            None,  # return_contribution
            None,  # instance_guided_geometry
        )
