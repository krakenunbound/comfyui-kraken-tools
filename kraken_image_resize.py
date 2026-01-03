import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import numpy as np
import folder_paths
import os

class KrakenImageResize:
    """
    Kraken Image Resize - A comprehensive image resizing node for ComfyUI
    Supports both image upload and chaining from upstream nodes
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available interpolation methods
        interpolation_methods = [
            "lanczos",
            "bicubic", 
            "bilinear",
            "nearest",
            "area"
        ]
        
        # Get list of images from ComfyUI input folder
        input_dir = folder_paths.get_input_directory()
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'))]
        
        return {
            "required": {
                "source_mode": (["upload", "upstream"], {"default": "upload"}),
                "image_upload": (sorted(image_files), {"image_upload": True}),
                "longest_side": ("INT", {
                    "default": 1024, 
                    "min": 64, 
                    "max": 8192, 
                    "step": 64,
                    "display": "number"
                }),
                "interpolation": (interpolation_methods, {"default": "lanczos"}),
                "maintain_aspect": ("BOOLEAN", {"default": True}),
                "resize_mode": (["longest_side", "width", "height", "fit", "fill", "stretch"], 
                              {"default": "longest_side"}),
                "upscale_smaller": ("BOOLEAN", {"default": False}),
                "output_format": (["same", "PNG", "JPEG", "WEBP"], {"default": "same"}),
            },
            "optional": {
                "image_input": ("IMAGE",),
                "width_override": ("INT", {
                    "default": 512, 
                    "min": 64, 
                    "max": 8192, 
                    "step": 64
                }),
                "height_override": ("INT", {
                    "default": 512, 
                    "min": 64, 
                    "max": 8192, 
                    "step": 64
                }),
                "background_color": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "info")
    FUNCTION = "resize_image"
    CATEGORY = "Kraken/Image"

    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def get_interpolation_method(self, method_name):
        """Get PIL interpolation method from string"""
        methods = {
            "lanczos": Image.Resampling.LANCZOS,
            "bicubic": Image.Resampling.BICUBIC,
            "bilinear": Image.Resampling.BILINEAR,
            "nearest": Image.Resampling.NEAREST,
            "area": Image.Resampling.BOX  # Closest to area interpolation
        }
        return methods.get(method_name, Image.Resampling.LANCZOS)

    def calculate_dimensions(self, original_width, original_height, longest_side, 
                           resize_mode, width_override, height_override, maintain_aspect):
        """Calculate target dimensions based on resize mode"""
        
        if resize_mode == "longest_side":
            if maintain_aspect:
                if original_width > original_height:
                    new_width = longest_side
                    new_height = int((original_height * longest_side) / original_width)
                else:
                    new_height = longest_side
                    new_width = int((original_width * longest_side) / original_height)
            else:
                new_width = new_height = longest_side
                
        elif resize_mode == "width":
            new_width = width_override or longest_side
            if maintain_aspect:
                new_height = int((original_height * new_width) / original_width)
            else:
                new_height = height_override or longest_side
                
        elif resize_mode == "height":
            new_height = height_override or longest_side
            if maintain_aspect:
                new_width = int((original_width * new_height) / original_height)
            else:
                new_width = width_override or longest_side
                
        elif resize_mode == "fit":
            # Fit within bounds maintaining aspect ratio
            target_w = width_override or longest_side
            target_h = height_override or longest_side
            ratio = min(target_w / original_width, target_h / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
        elif resize_mode == "fill":
            # Fill bounds maintaining aspect ratio (may crop)
            target_w = width_override or longest_side
            target_h = height_override or longest_side
            ratio = max(target_w / original_width, target_h / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
        elif resize_mode == "stretch":
            # Stretch to exact dimensions
            new_width = width_override or longest_side
            new_height = height_override or longest_side
            
        return new_width, new_height

    def resize_image(self, source_mode, image_upload, longest_side, interpolation, 
                    maintain_aspect, resize_mode, upscale_smaller, output_format,
                    image_input=None, width_override=None, height_override=None, 
                    background_color="#000000"):
        
        # Determine image source
        if source_mode == "upstream" and image_input is not None:
            # Use upstream image
            # Convert from ComfyUI tensor format to PIL
            image_tensor = image_input[0]  # Get first image from batch
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            source_info = "Upstream"
        else:
            # Load uploaded image
            input_dir = folder_paths.get_input_directory()
            image_path = os.path.join(input_dir, image_upload)
            pil_image = Image.open(image_path)
            source_info = f"Upload: {image_upload}"

        # Handle different image modes
        if pil_image.mode not in ('RGB', 'RGBA'):
            pil_image = pil_image.convert('RGB')

        original_width, original_height = pil_image.size
        
        # Calculate new dimensions
        new_width, new_height = self.calculate_dimensions(
            original_width, original_height, longest_side, 
            resize_mode, width_override, height_override, maintain_aspect
        )
        
        # Check if we should upscale smaller images
        original_size = max(original_width, original_height)
        target_size = max(new_width, new_height)
        
        if not upscale_smaller and target_size > original_size:
            new_width, new_height = original_width, original_height
            resize_info = "Skipped (would upscale)"
        else:
            # Perform the resize
            interpolation_method = self.get_interpolation_method(interpolation)
            
            if resize_mode == "fill":
                # For fill mode, we need to resize then crop/pad to exact dimensions
                target_w = width_override or longest_side
                target_h = height_override or longest_side
                
                # Create background
                bg_color = self.hex_to_rgb(background_color)
                background = Image.new('RGB', (target_w, target_h), bg_color)
                
                # Resize image
                resized = pil_image.resize((new_width, new_height), interpolation_method)
                
                # Center the resized image on background
                x_offset = (target_w - new_width) // 2
                y_offset = (target_h - new_height) // 2
                
                if resized.mode == 'RGBA':
                    background.paste(resized, (x_offset, y_offset), resized)
                else:
                    background.paste(resized, (x_offset, y_offset))
                
                pil_image = background
                new_width, new_height = target_w, target_h
            else:
                pil_image = pil_image.resize((new_width, new_height), interpolation_method)
            
            resize_info = f"Resized {original_width}x{original_height} → {new_width}x{new_height}"
        
        # Convert back to ComfyUI tensor format
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # Add batch dimension
        
        # Create info string
        info = f"{source_info} | {resize_info} | Method: {interpolation} | Mode: {resize_mode}"
        
        return (image_tensor, new_width, new_height, info)

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "KrakenImageResize": KrakenImageResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KrakenImageResize": "Kraken Image Resize"
}