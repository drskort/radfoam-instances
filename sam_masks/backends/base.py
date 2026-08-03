"""The interface both model adapters implement.

This is the only seam where model APIs appear. Everything else in the package is
written against this contract, which is what allowed the harness to be built and
tested before SAM 3.1's undocumented video API was known.
"""

from abc import ABC, abstractmethod


class Session(ABC):
    """A stateful video-tracking session over one frame sequence."""

    @abstractmethod
    def add_masks(self, frame_idx, masks):
        """Register masks as new tracked objects on frame_idx.

        masks is (N, H, W) bool. Returns a list of N object ids, unique within
        this session and never reused.
        """

    @abstractmethod
    def propagate(self):
        """Yield (frame_idx, {obj_id: (H, W) bool mask}) for every frame.

        Only LIVE objects appear: an object whose track was lost on a frame must
        be absent from that frame's dict, never present with an empty mask. The
        two backends disagree natively -- SAM 3.1 drops zero-area masks while
        SAM 2 returns a zero-filled row for every registered object -- so an
        adapter over a model that pads must filter. Without this, object counts
        would not be commensurable across arms and a model that lost every track
        would score as perfectly stable.
        """

    @abstractmethod
    def close(self):
        """Release model state and GPU memory."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


class Backend(ABC):
    """A segmentation model exposing everything-mode proposals and tracking."""

    name = None

    @abstractmethod
    def propose_masks(self, image, image_path=None):
        """Everything-mode proposals for one image.

        image is (H, W, 3) uint8 RGB. Returns (masks, scores) where masks is
        (N, H, W) bool, already filtered and deduplicated.

        image_path is the file the array came from, when the caller has it.
        SAM 3.1 cannot segment an in-memory array — build_sam3_image_model
        cannot load the 3.1 checkpoint, so its per-image arm runs the video
        predictor over a one-frame directory. Given image_path it symlinks the
        original instead of re-encoding a JPEG. Backends that do not need it
        ignore it.
        """

    @abstractmethod
    def start_session(self, frames_dir):
        """Open a Session over a directory of numbered frames."""
