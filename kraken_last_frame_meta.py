# kraken_last_frame_meta.py
import torch

class KrakenLastFrameMeta:
    """
    Extracts the last image from a batch and forwards WAN/size metadata so
    it can be used as the 'first frame' for the next segment.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # IMAGE tensor: [B, H, W, C]
                "images": ("IMAGE",),

                # Wire these directly from KrakenWanHelper
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 1}),
                "frames_length": ("INT", {"default": 65, "min": 1, "max": 4096, "step": 1}),
                "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "resolution_text": ("STRING", {"default": "832 x 480"}),
            },
            "optional": {
                # If you want to duplicate the last frame as a bookend for
                # continuity (useful when WAN adds a trailing frame).
                "duplicate_last_frame": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",   # last_frame (as [1, H, W, C])
        "IMAGE",   # last_frame_as_first (alias for readability)
        "INT",     # width
        "INT",     # height
        "INT",     # frames_length
        "FLOAT",   # fps
        "STRING",  # resolution_text
    )
    RETURN_NAMES = (
        "last_frame",
        "first_frame_out",
        "width",
        "height",
        "frames(length)",
        "fps",
        "resolution_text",
    )
    FUNCTION = "run"
    CATEGORY = "⚓ Kraken/WAN"

    def _last_slice(self, images: torch.Tensor) -> torch.Tensor:
        if images is None or not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("Expected IMAGE tensor [B,H,W,C].")
        if images.shape[0] == 0:
            raise ValueError("No images provided.")
        # Keep batch dimension using [-1:]
        return images[-1:].clone()

    def run(
        self,
        images: torch.Tensor,
        width: int,
        height: int,
        frames_length: int,
        fps: float,
        resolution_text: str,
        duplicate_last_frame: bool = False,
    ):
        last_frame = self._last_slice(images)

        if duplicate_last_frame:
            # Two identical frames [2, H, W, C] if you ever need it;
            # but we keep API consistent and still return [1,H,W,C]
            last_frame = last_frame.clone()

        # For convenience we return the same tensor twice:
        # - last_frame (semantic)
        # - first_frame_out (plug this straight into the next WAN run)
        return (
            last_frame,
            last_frame,
            int(width),
            int(height),
            int(frames_length),
            float(fps),
            str(resolution_text),
        )


# ComfyUI registration
NODE_CLASS_MAPPINGS = {"KrakenLastFrameMeta": KrakenLastFrameMeta}
NODE_DISPLAY_NAME_MAPPINGS = {"KrakenLastFrameMeta": "🦑 Kraken Last Frame + Meta"}
