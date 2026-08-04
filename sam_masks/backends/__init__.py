from sam_masks.backends.base import Backend, Session

__all__ = ["Backend", "Session", "get_backend"]


def get_backend(name, **kwargs):
    """Instantiate a backend by name; imports lazily so one model's missing
    dependencies cannot break the other arm."""
    if name == "sam21":
        from sam_masks.backends.sam21 import Sam21Backend

        return Sam21Backend(**kwargs)
    if name == "sam21_levels":
        # Same model, but SAM's subpart/part/whole decoder outputs are kept as
        # separate supervision levels instead of being flattened together.
        from sam_masks.backends.sam21 import Sam21Backend

        return Sam21Backend(levels=True, **kwargs)
    if name == "sam1":
        # The model OpenSplat3D's reference implementation uses. Image-only.
        from sam_masks.backends.sam1 import Sam1Backend

        return Sam1Backend(**kwargs)
    if name == "sam31":
        from sam_masks.backends.sam31 import Sam31Backend

        return Sam31Backend(**kwargs)
    raise ValueError(
        f"Unknown backend {name!r}; expected one of "
        "'sam1', 'sam21', 'sam21_levels', 'sam31'"
    )
