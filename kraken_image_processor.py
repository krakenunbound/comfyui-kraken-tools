import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import folder_paths
import os
import cv2
from scipy import ndimage

class KrakenImageProcessor:
    """
    🦑 Kraken Image Processor 🦑
    
    A versatile node for pre- and post-processing images in an upscaling pipeline.
    Pre-processing optimizes images for upscaling (denoising, contrast, sharpening).
    Post-processing refines upscaled images with subtle adjustments and optional film grain.
    
    Features:
    - Load images via upload or upstream input with explicit source mode
    - Denoising (bilateral, Gaussian, median, non-local means, torch-based)
    - Contrast/brightness optimization
    - Color correction (saturation, gamma)
    - Sharpening (unsharp mask, edge enhance, custom kernel)
    - Film grain for post-processing (artistic texture)
    - Quality metrics to evaluate changes
    - Alpha channel preservation
    - Robust fallback: runs even with no upstream tensor or uploaded file
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        image_files = [f for f in os.listdir(input_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'))]
        # Always include a safe "(none)" choice so the UI doesn't force a real file
        choices = ["(none)"] + sorted(image_files)

        return {
            "required": {
                # === MODE SELECTION ===
                "processing_mode": (["pre_process", "post_process"], {
                    "default": "pre_process",
                    "tooltip": "Pre-process: Prepare image for upscaling | Post-process: Refine upscaled image"
                }),
                "source_mode": (["upload", "upstream"], {
                    "default": "upstream",  # prefer upstream by default
                    "tooltip": "Upload: use uploaded image | Upstream: receive from connected node"
                }),
                "image_upload": (choices, {
                    "image_upload": True,
                    "tooltip": "Select uploaded image (only used when source_mode = upload)"
                }),
                
                # === NOISE REDUCTION SECTION ===
                "enable_denoising": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove noise/artifacts (recommended for pre-processing)"
                }),
                "denoise_method": (["bilateral", "gaussian", "gaussian_torch", "median", "non_local_means"], {
                    "default": "bilateral",
                    "tooltip": "bilateral (edge-preserving) | gaussian (fast) | gaussian_torch (GPU) | median (JPEG artifacts) | non_local_means (heavy)"
                }),
                "denoise_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Denoising intensity (1.0 = normal, higher = aggressive)"
                }),
                
                # === CONTRAST & BRIGHTNESS OPTIMIZATION ===
                "enable_contrast_enhancement": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Optimize contrast and brightness for detail visibility"
                }),
                "auto_contrast": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically optimize contrast range"
                }),
                "contrast_factor": ("FLOAT", {
                    "default": 1.05,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Manual contrast adjustment (1.0 = no change)"
                }),
                "brightness_factor": ("FLOAT", {
                    "default": 1.10,
                    "min": 0.5,
                    "max": 1.5,
                    "step": 0.01,
                    "tooltip": "Brightness adjustment (1.0 = no change)"
                }),
                
                # === SHARPENING SECTION ===
                "enable_pre_sharpening": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Sharpen edges (pre: before upscale, post: subtle refinement)"
                }),
                "sharpening_method": (["unsharp_mask", "edge_enhance", "custom_kernel"], {
                    "default": "unsharp_mask",
                    "tooltip": "unsharp_mask (professional) | edge_enhance (simple) | custom_kernel (advanced)"
                }),
                "sharpening_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Sharpening intensity (0.3 = subtle, 1.0 = strong)"
                }),
                "sharpening_radius": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Sharpening radius (smaller = fine detail, larger = broader)"
                }),
                
                # === COLOR OPTIMIZATION ===
                "enable_color_correction": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply color corrections (useful for faded/poor colors)"
                }),
                "color_saturation": ("FLOAT", {
                    "default": 1.05,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Color saturation (1.0 = normal, >1.0 = vibrant)"
                }),
                "gamma_correction": ("FLOAT", {
                    "default": 1.10,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Gamma correction (1.0 = normal, <1.0 = brighter)"
                }),
                
                # === FILM GRAIN SECTION (POST-PROCESS ONLY) ===
                "enable_film_grain": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add film grain for artistic texture (post-processing only)"
                }),
                "grain_amount": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Grain intensity (0.05 = subtle, 0.5 = strong)"
                }),
                "grain_size": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Grain size (1.0 = fine, 3.0 = coarse)"
                }),
                
                # === ADVANCED OPTIONS ===
                "preserve_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve transparency channel if present"
                }),
                "processing_precision": (["8bit", "16bit"], {
                    "default": "16bit",
                    "tooltip": "Processing bit depth (16bit = higher quality, slower)"
                }),
            },
            "optional": {
                "image_input": ("IMAGE", {
                    "tooltip": "Input image from upstream node (used if source_mode = upstream)"
                }),
                "custom_sharpen_kernel": ("STRING", {
                    "default": "0,-1,0;-1,5,-1;0,-1,0",
                    "tooltip": "Custom sharpening kernel (format: row1;row2;row3)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("processed_image", "processing_log", "quality_metrics")
    FUNCTION = "process_image"
    CATEGORY = "Kraken/Image"

    @classmethod
    def VALIDATE_INPUTS(cls, source_mode, image_upload, **kwargs):
        """Validate inputs, especially uploaded file paths for security"""
        if source_mode == "upload" and image_upload and image_upload != "(none)":
            if not folder_paths.exists_annotated_filepath(image_upload):
                return f"Invalid image file: {image_upload}"
        return True

    # --- NEW: Placeholder factory so the node always has something to process ---
    def create_placeholder_image(self, preserve_alpha=True, w=512, h=512):
        mode = 'RGBA' if preserve_alpha else 'RGB'
        if mode == 'RGBA':
            return Image.new('RGBA', (w, h), (0, 0, 0, 0))  # transparent
        return Image.new('RGB', (w, h), (0, 0, 0))  # black

    def tensor_to_pil(self, tensor, preserve_alpha_in_tensor=False):
        """Convert ComfyUI image tensor to PIL Image"""
        if len(tensor.shape) == 4:
            if preserve_alpha_in_tensor and tensor.shape[3] == 4:
                image_np = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
                return Image.fromarray(image_np, 'RGBA')
            tensor = tensor[0]
        image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor format, preserving alpha"""
        if pil_image.mode not in ('RGB', 'RGBA', 'L'):
             pil_image = pil_image.convert('RGB')
        
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0)

    def pil_to_cv2(self, pil_image):
        """Convert PIL Image to OpenCV format"""
        # This function assumes an RGB PIL image
        np_image = np.array(pil_image)
        if len(np_image.shape) == 3 and np_image.shape[2] == 3:
            return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        return np_image  # Return as is if not 3-channel RGB

    def cv2_to_pil(self, cv2_image):
        """Convert OpenCV image to PIL format"""
        if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
            return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        return Image.fromarray(cv2_image)  # Return as is if not 3-channel BGR

    def apply_denoising(self, pil_image, method, strength, processing_precision):
        """Apply noise reduction with various methods on an RGB image"""
        dtype = np.float16 if processing_precision == "16bit" else np.float32
        
        if method == "gaussian_torch" and torch.cuda.is_available():
            tensor = self.pil_to_tensor(pil_image)
            tensor_chw = tensor.permute(0, 3, 1, 2)
            kernel_size = int(3 * strength) | 1
            kernel = torch.ones(3, 1, kernel_size, kernel_size, device=tensor.device) / (kernel_size ** 2)
            blurred_chw = torch.nn.functional.conv2d(tensor_chw, kernel, padding=kernel_size//2, groups=3)
            blurred_hwc = blurred_chw.permute(0, 2, 3, 1)
            return self.tensor_to_pil(blurred_hwc)
        
        cv2_image = self.pil_to_cv2(pil_image)
        if method == "bilateral":
            d = int(5 * strength)
            sigma_color = 80 * strength
            sigma_space = 80 * strength
            denoised = cv2.bilateralFilter(cv2_image, d, sigma_color, sigma_space)
        elif method == "gaussian":
            kernel_size = int(3 * strength) | 1
            denoised = cv2.GaussianBlur(cv2_image, (kernel_size, kernel_size), strength)
        elif method == "median":
            kernel_size = int(3 * strength) | 1
            denoised = cv2.medianBlur(cv2_image, kernel_size)
        elif method == "non_local_means":
            h_val = 10 * strength
            denoised = cv2.fastNlMeansDenoisingColored(
                cv2_image, None, h=h_val, hColor=h_val, 
                templateWindowSize=7, searchWindowSize=21
            )
        else:
            denoised = cv2_image
        return self.cv2_to_pil(denoised)

    def apply_contrast_enhancement(self, pil_image, auto_contrast, contrast_factor, brightness_factor):
        """Enhance contrast and brightness on an RGB image"""
        enhanced = pil_image
        if auto_contrast:
            enhanced = ImageOps.autocontrast(enhanced, cutoff=1)
        if contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast_factor)
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness_factor)
        return enhanced

    def apply_sharpening(self, pil_image, method, strength, radius, custom_kernel, processing_precision):
        """Apply sharpening to enhance edges on an RGB image"""
        if strength <= 0:
            return pil_image
        dtype = np.float16 if processing_precision == "16bit" else np.float32
        
        if method == "unsharp_mask":
            original_np = np.array(pil_image, dtype=dtype)
            blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
            blurred_np = np.array(blurred, dtype=dtype)
            mask = original_np - blurred_np
            sharpened_np = np.clip(original_np + (strength * mask), 0, 255)
            return Image.fromarray(sharpened_np.astype(np.uint8))
        elif method == "edge_enhance":
            enhancer = ImageEnhance.Sharpness(pil_image)
            return enhancer.enhance(1.0 + strength)
        elif method == "custom_kernel":
            try:
                rows = custom_kernel.split(';')
                kernel = [[float(x.strip()) for x in row.split(',')] for row in rows]
                kernel_np = np.array(kernel, dtype=dtype)
                cv2_image = self.pil_to_cv2(pil_image)
                sharpened = cv2.filter2D(cv2_image, -1, kernel_np)
                return self.cv2_to_pil(sharpened)
            except Exception:
                enhancer = ImageEnhance.Sharpness(pil_image)
                return enhancer.enhance(1.0 + strength)
        return pil_image

    def apply_color_correction(self, pil_image, saturation, gamma, processing_precision):
        """Apply color corrections on an RGB image"""
        dtype = np.float16 if processing_precision == "16bit" else np.float32
        enhanced = pil_image
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(saturation)
        if gamma != 1.0:
            np_image = np.array(enhanced, dtype=dtype) / 255.0
            gamma_corrected = np.power(np_image, 1.0 / gamma)
            enhanced = Image.fromarray((np.clip(gamma_corrected * 255, 0, 255)).astype(np.uint8))
        return enhanced

    def apply_film_grain(self, pil_image, amount, size, processing_precision):
        """Add film grain for artistic texture on an RGB image"""
        if amount <= 0:
            return pil_image
        dtype = np.float16 if processing_precision == "16bit" else np.float32
        np_image = np.array(pil_image, dtype=dtype)
        h, w, c = np_image.shape
        noise_intensity = amount * 255
        noise = np.random.normal(0, noise_intensity, (int(h / size), int(w / size), c))
        if size != 1.0:
            noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        noisy_image = np.clip(np_image + noise, 0, 255)
        return Image.fromarray(noisy_image.astype(np.uint8))

    def calculate_quality_metrics(self, original_pil, processed_pil):
        """Calculate quality metrics to evaluate changes"""
        orig_gray = original_pil.convert('L')
        proc_gray = processed_pil.convert('L')
        orig_np = np.array(orig_gray, dtype=np.float32)
        proc_np = np.array(proc_gray, dtype=np.float32)
        
        orig_std = np.std(orig_np)
        proc_std = np.std(proc_np)
        orig_mean = np.mean(orig_np)
        proc_mean = np.mean(proc_np)
        
        orig_edges = cv2.Canny(orig_np.astype(np.uint8), 50, 150)
        proc_edges = cv2.Canny(proc_np.astype(np.uint8), 50, 150)
        orig_edge_density = np.sum(orig_edges > 0) / orig_edges.size if orig_edges.size > 0 else 0
        proc_edge_density = np.sum(proc_edges > 0) / proc_edges.size if proc_edges.size > 0 else 0
        
        metrics = {
            "contrast_change": proc_std / orig_std if orig_std > 0 else 1.0,
            "brightness_change": proc_mean - orig_mean,
            "edge_density_change": proc_edge_density / orig_edge_density if orig_edge_density > 0 else 1.0,
        }
        
        quality_report = (
            f"Contrast: {metrics['contrast_change']:.2f}x | "
            f"Brightness: {metrics['brightness_change']:+.1f} | "
            f"Edges: {metrics['edge_density_change']:.2f}x"
        )
        return quality_report

    def process_image(self, processing_mode, source_mode, image_upload, enable_denoising, denoise_method, 
                      denoise_strength, enable_contrast_enhancement, auto_contrast, contrast_factor, 
                      brightness_factor, enable_pre_sharpening, sharpening_method, sharpening_strength, 
                      sharpening_radius, enable_color_correction, color_saturation, gamma_correction,
                      enable_film_grain, grain_amount, grain_size, preserve_alpha, processing_precision,
                      image_input=None, custom_sharpen_kernel=None):
        """Main processing pipeline for pre- or post-processing"""
        processing_log = []
        
        # Clamp/adjust some params in post mode
        if processing_mode == "post_process":
            denoise_strength = min(denoise_strength, 0.5)
            sharpening_strength = min(sharpening_strength, 0.2)
            contrast_factor = min(contrast_factor, 1.1)
            brightness_factor = min(brightness_factor, 1.05)
            enable_denoising = enable_denoising and denoise_method != "non_local_means"
            auto_contrast = False
        else:
            enable_film_grain = False

        # --- Robust source selection (never crashes) ---
        pil_image = None
        source_info = ""

        if source_mode == "upstream":
            if image_input is not None:
                pil_image = self.tensor_to_pil(image_input, preserve_alpha_in_tensor=True)
                source_info = "Source: Upstream"
            else:
                # No upstream tensor — create placeholder
                pil_image = self.create_placeholder_image(preserve_alpha=preserve_alpha)
                source_info = "Source: Placeholder (no upstream image)"
        else:  # source_mode == "upload"
            use_placeholder = (image_upload is None) or (image_upload == "(none)")
            if not use_placeholder:
                # Use ComfyUI's built-in path sanitization to prevent path traversal attacks
                image_path = folder_paths.get_annotated_filepath(image_upload)
                try:
                    pil_image = Image.open(image_path)
                    pil_image = ImageOps.exif_transpose(pil_image)
                    source_info = f"Source: {os.path.basename(image_upload)}"
                except Exception:
                    pil_image = self.create_placeholder_image(preserve_alpha=preserve_alpha)
                    source_info = f"Source: Placeholder (failed to open '{os.path.basename(image_upload)}')"
            else:
                pil_image = self.create_placeholder_image(preserve_alpha=preserve_alpha)
                source_info = "Source: Placeholder (no uploaded image)"

        original_pil = pil_image.copy()
        
        # --- ALPHA CHANNEL HANDLING: SEPARATE AT THE START ---
        has_alpha = pil_image.mode == 'RGBA'
        alpha_channel = None
        if has_alpha and preserve_alpha:
            alpha_channel = pil_image.split()[-1]
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        processing_log.append(source_info + f" ({pil_image.width}x{pil_image.height})")
        
        # --- PROCESSING PIPELINE (OPERATES ON RGB IMAGE) ---
        if enable_denoising:
            pil_image = self.apply_denoising(pil_image, denoise_method, denoise_strength, processing_precision)
            if alpha_channel:  # Also denoise alpha channel separately
                alpha_channel = alpha_channel.filter(ImageFilter.GaussianBlur(radius=denoise_strength * 0.5))
            processing_log.append(f"Denoise({denoise_method}:{denoise_strength})")
        
        if enable_contrast_enhancement:
            pil_image = self.apply_contrast_enhancement(pil_image, auto_contrast, contrast_factor, brightness_factor)
            processing_log.append(f"Contrast({contrast_factor})")

        if enable_color_correction:
            pil_image = self.apply_color_correction(pil_image, color_saturation, gamma_correction, processing_precision)
            processing_log.append(f"Color(sat:{color_saturation},gamma:{gamma_correction})")
        
        if enable_pre_sharpening:
            pil_image = self.apply_sharpening(
                pil_image, sharpening_method, sharpening_strength, sharpening_radius, 
                custom_sharpen_kernel, processing_precision
            )
            processing_log.append(f"Sharpen({sharpening_method}:{sharpening_strength})")
        
        if enable_film_grain and processing_mode == "post_process":
            pil_image = self.apply_film_grain(pil_image, grain_amount, grain_size, processing_precision)
            processing_log.append(f"Grain({grain_amount})")
        
        # --- ALPHA CHANNEL HANDLING: MERGE AT THE END ---
        if has_alpha and preserve_alpha and alpha_channel:
            pil_image.putalpha(alpha_channel)
            processing_log.append("Alpha Restored")
        
        quality_report = self.calculate_quality_metrics(original_pil.convert('RGB'), pil_image.convert('RGB'))
        
        output_tensor = self.pil_to_tensor(pil_image)
        processing_summary = " → ".join(processing_log)
        
        return (output_tensor, processing_summary, quality_report)

NODE_CLASS_MAPPINGS = {
    "KrakenImageProcessor": KrakenImageProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KrakenImageProcessor": "🦑 Kraken Image Processor"
}
