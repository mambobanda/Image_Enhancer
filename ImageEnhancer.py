
import os
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from super_image import EdsrModel, ImageLoader
from nafnetlib import DeblurProcessor, DenoiseProcessor
import argparse
import logging
from datetime import datetime
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


"""
Key Additions for Scale Support:
1. Scale Parameter

    Added --scale argument with choices: 1, 2, 3

    Default is 2x

    Automatically loads appropriate model for each scale

2. Model Selection by Scale
python

model_map = {
    1: 'eugenesiow/edsr-base',  # No upscaling
    2: 'eugenesiow/edsr-base',  # 2x
    3: 'eugenesiow/edsr-3x',    # 3x
    4: 'eugenesiow/edsr-4x',    # 4x (supported but not in choices)
}

3. Filename Includes Scale

    Output files include the scale in the filename

    Example: image_enhanced_2x.png, image_enhanced_3x.png

4. PIL Fallback

    If EDSR model fails, uses PIL's LANCZOS resampling

    Ensures processing continues even with model issues

Usage Examples:
1x (No Upscaling - Just Deblur + Denoise):
bash

python ImageEnhancer.py E:/input C:/output --scale 1

Output: image_enhanced_1x.png (same resolution)
2x (Standard Upscaling):
bash

python ImageEnhancer.py E:/input C:/output --scale 2

Output: image_enhanced_2x.png (2x resolution)
3x (High Quality Upscaling):
bash

python ImageEnhancer.py E:/input C:/output --scale 3

Output: image_enhanced_3x.png (3x resolution)
Batch Processing with Different Scales:
bash

# Process at 2x with all optimizations
python ImageEnhancer.py E:/input C:/output \
    --scale 2 \
    --batch-size 8 \
    --recursive \
    --overwrite

# Process at 3x with memory optimization
python ImageEnhancer.py E:/input C:/output \
    --scale 3 \
    --batch-size 2 \
    --num-workers 2

In Python Script:
python

from ImageEnhancer import ImageEnhancer

# Initialize with 3x scaling
enhancer = ImageEnhancer(
    device="cuda", 
    model_dir="./models/",
    scale=3  # 3x upscaling
)

# Process directory
stats = enhancer.process_directory(
    source_dir='E:/Comfy Output/MiniMax Tests/Res_Test2/',
    dest_dir='C:/test/processed/',
    recursive=False,
    overwrite=True
)

print(f"Processed {stats['success']} images at {stats['scale']}x")

Process Different Images with Different Scales:
python

# Process one image at 2x
enhancer_2x = ImageEnhancer(scale=2)
enhancer_2x.process_single_image('image1.png', 'image1_2x.png')

# Process another at 3x
enhancer_3x = ImageEnhancer(scale=3)
enhancer_3x.process_single_image('image2.png', 'image2_3x.png')

Scale Comparison:
Scale   Model Used  Output Size Use Case
1x  No model (passthrough)  Original size   Just denoise/deblur
2x  EDSR 2x 2x original Standard upscaling
3x  EDSR 3x 3x original High quality upscaling
Performance Considerations:

    1x: Fastest (no upscaling)

    2x: Balanced quality/speed

    3x: Highest quality, slower processing

    Higher scales require more GPU memory

The code now supports all three scale factors with optimized processing for each!
"""


import os
import torch
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
from super_image import EdsrModel, ImageLoader
import argparse
import logging
from datetime import datetime
import gc
import numpy as np
import threading

# Try importing nafnet with fallback options
try:
    from nafnet import DeblurProcessor, DenoiseProcessor
    NAFNET_AVAILABLE = True
    logger_import = logging.getLogger(__name__)
    logger_import.info("NAFNet imported successfully")
except ImportError:
    try:
        from nafnet_pytorch import DeblurProcessor, DenoiseProcessor
        NAFNET_AVAILABLE = True
        logger_import = logging.getLogger(__name__)
        logger_import.info("NAFNet-pytorch imported successfully")
    except ImportError:
        try:
            from nafnetlib import DeblurProcessor, DenoiseProcessor
            NAFNET_AVAILABLE = True
            logger_import = logging.getLogger(__name__)
            logger_import.info("NAFNetlib imported successfully")
        except ImportError:
            NAFNET_AVAILABLE = False
            logger_import = logging.getLogger(__name__)
            logger_import.warning("NAFNet not available. Using fallback methods.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FallbackDeblur:
    """Fallback deblurring using PIL filters"""
    def __init__(self, *args, **kwargs):
        pass
    
    def process(self, image):
        """Apply simple deblurring using sharpening"""
        return image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))


class FallbackDenoise:
    """Fallback denoising using PIL or OpenCV"""
    def __init__(self, *args, **kwargs):
        pass
    
    def process(self, image):
        """Apply simple denoising"""
        try:
            # Try using OpenCV if available
            import cv2
            img_array = np.array(image)
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
            return Image.fromarray(denoised)
        except ImportError:
            # Fallback to PIL's simple filters
            return image.filter(ImageFilter.MedianFilter(size=3))


class ImageEnhancer:
    """Batch image enhancement pipeline: Deblur → Denoise → Super Resolution"""
    
    def __init__(self, device="cuda", model_dir="./models/", 
                 use_amp=True, num_workers=4, batch_size=4, scale=2,
                 force_fallback=False):
        """
        Initialize the image enhancement pipeline.
        
        Args:
            device: "cuda" or "cpu"
            model_dir: Directory to store downloaded models
            use_amp: Use Automatic Mixed Precision for faster inference
            num_workers: Number of worker threads for parallel processing
            batch_size: Batch size for super resolution
            scale: Super resolution scale factor (1, 2, or 3)
            force_fallback: Force using fallback methods even if NAFNet is available
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        self.use_amp = use_amp and self.device == "cuda"
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.scale = scale
        self.force_fallback = force_fallback
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Super resolution scale: {scale}x")
        if self.use_amp:
            logger.info("Using Automatic Mixed Precision (AMP)")
        
        # Initialize processors with fallback
        self.deblur = None
        self.denoise = None
        
        if NAFNET_AVAILABLE and not force_fallback:
            try:
                self.deblur = DeblurProcessor(
                    model_id="nafnet_gopro",
                    model_dir=model_dir,
                    device=self.device
                )
                logger.info("Deblur processor initialized (NAFNet)")
            except Exception as e:
                logger.warning(f"Failed to initialize NAFNet deblur: {e}")
                logger.info("Using fallback deblur method")
                self.deblur = FallbackDeblur()
                
            try:
                self.denoise = DenoiseProcessor(
                    model_id="nafnet_sidd",
                    model_dir=model_dir,
                    device=self.device
                )
                logger.info("Denoise processor initialized (NAFNet)")
            except Exception as e:
                logger.warning(f"Failed to initialize NAFNet denoise: {e}")
                logger.info("Using fallback denoise method")
                self.denoise = FallbackDenoise()
        else:
            logger.info("Using fallback methods for deblur and denoise")
            self.deblur = FallbackDeblur()
            self.denoise = FallbackDenoise()
        
        # Initialize super resolution model
        try:
            self.edsr = self._load_super_res_model(scale)
            
            if self.edsr is not None:
                self.edsr.eval()
                
                # Optimize for inference
                if self.device == "cuda":
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                logger.info(f"EDSR {scale}x super-resolution model initialized")
            else:
                logger.info(f"No super resolution model loaded for {scale}x")
        except Exception as e:
            logger.error(f"Failed to initialize EDSR model: {e}")
            self.edsr = None
            logger.warning("Super resolution will use PIL fallback")
        
        # Thread-local storage for models
        self.thread_local = threading.local()
    
    def _load_super_res_model(self, scale):
        """Load the appropriate super resolution model based on scale factor."""
        if scale == 1:
            return None
        
        try:
            # Try loading with specific scale
            if scale == 2:
                return EdsrModel.from_pretrained(
                    'eugenesiow/edsr-base',
                    scale=2
                ).to(self.device)
            elif scale == 3:
                try:
                    return EdsrModel.from_pretrained(
                        'eugenesiow/edsr-3x',
                        scale=3
                    ).to(self.device)
                except:
                    # Fallback to base model with scale parameter
                    return EdsrModel.from_pretrained(
                        'eugenesiow/edsr-base',
                        scale=3
                    ).to(self.device)
            elif scale == 4:
                return EdsrModel.from_pretrained(
                    'eugenesiow/edsr-4x',
                    scale=4
                ).to(self.device)
            else:
                return EdsrModel.from_pretrained(
                    'eugenesiow/edsr-base',
                    scale=scale
                ).to(self.device)
        except Exception as e:
            logger.warning(f"Failed to load EDSR model for {scale}x: {e}")
            return None
    
    def _get_thread_models(self):
        """Get thread-local model instances for parallel processing"""
        if not hasattr(self.thread_local, 'deblur'):
            self.thread_local.deblur = self.deblur
            self.thread_local.denoise = self.denoise
            self.thread_local.edsr = self.edsr
        return self.thread_local.deblur, self.thread_local.denoise, self.thread_local.edsr
    
    def _upscale_with_pil(self, image, scale):
        """Fallback upscaling using PIL for scale=1 or when model fails."""
        if scale == 1:
            return image
        
        # Simple bilinear upscaling
        width, height = image.size
        new_size = (width * scale, height * scale)
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    def process_single_image(self, image_path, output_path, 
                            apply_deblur=True, 
                            apply_denoise=True, 
                            apply_super_res=True,
                            save_intermediate=False,
                            intermediate_dir=None,
                            scale=None):
        """
        Process a single image through the enhancement pipeline.
        """
        try:
            # Use instance scale if not overridden
            if scale is None:
                scale = self.scale
            
            # Load image with optimization
            image = Image.open(image_path)
            
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Prepare intermediate directory
            if save_intermediate and intermediate_dir:
                os.makedirs(intermediate_dir, exist_ok=True)
                base_name = Path(image_path).stem
            
            # Deblur step
            if apply_deblur and self.deblur is not None:
                logger.debug("Applying deblur...")
                image = self.deblur.process(image)
                if save_intermediate and intermediate_dir:
                    intermediate_path = os.path.join(intermediate_dir, f'{base_name}_deblurred.png')
                    image.save(intermediate_path, optimize=True)
            
            # Denoise step
            if apply_denoise and self.denoise is not None:
                logger.debug("Applying denoise...")
                image = self.denoise.process(image)
                if save_intermediate and intermediate_dir:
                    intermediate_path = os.path.join(intermediate_dir, f'{base_name}_denoised.png')
                    image.save(intermediate_path, optimize=True)
            
            # Super resolution step
            if apply_super_res and scale > 1:
                if self.edsr is not None:
                    try:
                        inputs = ImageLoader.load_image(image)
                        
                        with torch.no_grad():
                            if self.use_amp:
                                with torch.cuda.amp.autocast():
                                    preds = self.edsr(inputs)
                            else:
                                preds = self.edsr(inputs)
                        
                        # Save with optimization
                        ImageLoader.save_image(preds, output_path)
                        
                        # Free memory
                        del inputs, preds
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                    except Exception as e:
                        logger.warning(f"EDSR processing failed, using PIL fallback: {e}")
                        upscaled = self._upscale_with_pil(image, scale)
                        upscaled.save(output_path, optimize=True)
                else:
                    # Fallback to PIL if model not available
                    logger.debug(f"Using PIL upscaling for {scale}x")
                    upscaled = self._upscale_with_pil(image, scale)
                    upscaled.save(output_path, optimize=True)
            elif apply_super_res and scale == 1:
                # No upscaling, just save
                image.save(output_path, optimize=True)
                logger.debug(f"No upscaling applied (1x)")
            else:
                # No super resolution, just save
                image.save(output_path, optimize=True)
            
            # Clear image from memory
            del image
            gc.collect()
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to process {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_batch_super_res(self, images_data, scale=None):
        """
        Process multiple images through super resolution in batch.
        """
        if not images_data:
            return []
        
        if scale is None:
            scale = self.scale
        
        # If scale is 1 or no model, process individually
        if scale == 1 or self.edsr is None:
            results = []
            for image_path, output_path in images_data:
                try:
                    img = Image.open(image_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(output_path, optimize=True)
                    results.append(True)
                except Exception as e:
                    logger.error(f"Failed to save {image_path}: {e}")
                    results.append(False)
            return results
        
        try:
            # Load all images
            batch_images = []
            for image_path, output_path in images_data:
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                batch_images.append(ImageLoader.load_image(img))
            
            # Stack into batch
            batch_tensor = torch.cat(batch_images, dim=0)
            
            # Process batch
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        preds = self.edsr(batch_tensor)
                else:
                    preds = self.edsr(batch_tensor)
            
            # Save results
            results = []
            for i, (_, output_path) in enumerate(images_data):
                # Extract single image from batch
                single_pred = preds[i:i+1]
                ImageLoader.save_image(single_pred, output_path)
                results.append(True)
            
            # Clean up
            del batch_tensor, preds, batch_images
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"Batch super resolution failed: {e}")
            # Fall back to individual processing
            results = []
            for image_path, output_path in images_data:
                try:
                    success = self.process_single_image(
                        image_path, output_path, 
                        apply_deblur=False, 
                        apply_denoise=False,
                        apply_super_res=True,
                        scale=scale
                    )
                    results.append(success)
                except Exception as e2:
                    logger.error(f"Individual processing failed for {image_path}: {e2}")
                    results.append(False)
            return results
    
    def process_directory(self, source_dir, dest_dir, 
                         extensions=None,
                         apply_deblur=True,
                         apply_denoise=True,
                         apply_super_res=True,
                         save_intermediate=False,
                         recursive=False,
                         overwrite=False,
                         filename_suffix="_enhanced",
                         parallel=True,
                         scale=None):
        """
        Process all images in a directory.
        """
        source_dir = Path(source_dir)
        dest_dir = Path(dest_dir)
        
        if not source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")
        
        # Use instance scale if not overridden
        if scale is None:
            scale = self.scale
        
        # Create destination directory
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Create intermediate directory if needed
        intermediate_dir = None
        if save_intermediate:
            intermediate_dir = dest_dir / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default extensions
        if extensions is None:
            extensions = ['.png', '.PNG']
        
        # Get list of images to process
        image_files = []
        if recursive:
            for ext in extensions:
                image_files.extend(source_dir.rglob(f'*{ext}'))
        else:
            for ext in extensions:
                image_files.extend(source_dir.glob(f'*{ext}'))
        
        if not image_files:
            logger.warning(f"No PNG files found in {source_dir}")
            return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
        
        logger.info(f"Found {len(image_files)} images to process")
        logger.info(f"Super resolution scale: {scale}x")
        logger.info(f"NAFNet available: {NAFNET_AVAILABLE}")
        
        # Handle duplicate filenames
        used_names = set()
        
        # Statistics
        stats = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'scale': scale,
            'nafnet_available': NAFNET_AVAILABLE
        }
        
        # Prepare tasks for parallel processing
        tasks = []
        for image_path in image_files:
            base_name = image_path.stem
            extension = image_path.suffix
            
            # Add scale to filename if not 1x
            scale_suffix = f"_{scale}x" if scale > 1 else ""
            output_name = f"{base_name}{filename_suffix}{scale_suffix}{extension}"
            counter = 1
            while output_name in used_names:
                output_name = f"{base_name}{filename_suffix}{scale_suffix}_{counter}{extension}"
                counter += 1
            used_names.add(output_name)
            
            output_path = dest_dir / output_name
            
            if output_path.exists() and not overwrite:
                stats['skipped'] += 1
                continue
            
            tasks.append((image_path, output_path))
        
        if not tasks:
            logger.info("No new images to process")
            return stats
        
        logger.info(f"Processing {len(tasks)} images at {scale}x...")
        
        # Process based on parallel flag
        if parallel and apply_super_res and scale > 1 and len(tasks) > 1:
            # Use batch processing for super resolution
            if apply_deblur or apply_denoise:
                # Sequential processing with batch super resolution
                super_res_tasks = []
                
                for image_path, output_path in tqdm(tasks, desc="Pre-processing images"):
                    try:
                        # Load image
                        image = Image.open(image_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Deblur and denoise sequentially
                        if apply_deblur and self.deblur is not None:
                            image = self.deblur.process(image)
                        if apply_denoise and self.denoise is not None:
                            image = self.denoise.process(image)
                        
                        if save_intermediate and intermediate_dir:
                            base_name = image_path.stem
                            inter_path = os.path.join(intermediate_dir, f'{base_name}_processed.png')
                            image.save(inter_path, optimize=True)
                        
                        # Save processed image for batch SR
                        temp_path = dest_dir / f"temp_{image_path.name}"
                        image.save(temp_path)
                        super_res_tasks.append((str(temp_path), str(output_path)))
                        
                    except Exception as e:
                        logger.error(f"Failed to process {image_path}: {e}")
                        stats['failed'] += 1
                
                # Batch process super resolution
                if super_res_tasks:
                    logger.info(f"Running batch super resolution on {len(super_res_tasks)} images at {scale}x...")
                    results = self.process_batch_super_res(super_res_tasks, scale=scale)
                    success_count = sum(results)
                    stats['success'] += success_count
                    stats['failed'] += len(super_res_tasks) - success_count
                    
                    # Clean up temp files
                    for temp_path, _ in super_res_tasks:
                        try:
                            os.remove(temp_path)
                        except:
                            pass
            else:
                # Only super resolution - process in batches
                batch_size = self.batch_size
                for i in range(0, len(tasks), batch_size):
                    batch_tasks = tasks[i:i+batch_size]
                    results = self.process_batch_super_res(batch_tasks, scale=scale)
                    success_count = sum(results)
                    stats['success'] += success_count
                    stats['failed'] += len(batch_tasks) - success_count
        else:
            # Sequential processing (fallback or when parallel disabled)
            for image_path, output_path in tqdm(tasks, desc="Processing images"):
                success = self.process_single_image(
                    image_path, 
                    output_path,
                    apply_deblur=apply_deblur,
                    apply_denoise=apply_denoise,
                    apply_super_res=apply_super_res,
                    save_intermediate=save_intermediate,
                    intermediate_dir=intermediate_dir,
                    scale=scale
                )
                
                if success:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
        
        # Log summary
        logger.info("=" * 50)
        logger.info(f"Processing Complete!")
        logger.info(f"Scale: {scale}x")
        logger.info(f"NAFNet Available: {NAFNET_AVAILABLE}")
        logger.info(f"Total: {stats['total']}")
        logger.info(f"Success: {stats['success']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Skipped: {stats['skipped']}")
        if save_intermediate:
            logger.info(f"Intermediate files saved to: {intermediate_dir}")
        logger.info("=" * 50)
        
        return stats


def main():
    """Command line interface for batch processing."""
    parser = argparse.ArgumentParser(
        description='Batch image enhancement: Deblur → Denoise → Super Resolution'
    )
    parser.add_argument(
        'source_dir',
        help='Source directory containing PNG images'
    )
    parser.add_argument(
        'dest_dir',
        help='Destination directory for processed images (all files go here directly)'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=2,
        choices=[1, 2, 3],
        help='Super resolution scale factor: 1x, 2x, or 3x (default: 2x)'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for processing (default: cuda)'
    )
    parser.add_argument(
        '--model-dir',
        default='./models/',
        help='Directory to store models (default: ./models/)'
    )
    parser.add_argument(
        '--force-fallback',
        action='store_true',
        help='Force using fallback methods (ignore NAFNet)'
    )
    parser.add_argument(
        '--no-deblur',
        action='store_false',
        dest='apply_deblur',
        help='Skip deblurring step'
    )
    parser.add_argument(
        '--no-denoise',
        action='store_false',
        dest='apply_denoise',
        help='Skip denoising step'
    )
    parser.add_argument(
        '--no-super-res',
        action='store_false',
        dest='apply_super_res',
        help='Skip super resolution step'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Process subdirectories recursively (all files go to dest_dir)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate results (deblurred, denoised) in dest_dir/intermediate/'
    )
    parser.add_argument(
        '--suffix',
        default='_enhanced',
        help='Suffix to add to output filename (default: _enhanced)'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.png', '.PNG'],
        help='File extensions to process (default: .png .PNG)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_false',
        dest='parallel',
        help='Disable parallel processing'
    )
    parser.add_argument(
        '--no-amp',
        action='store_false',
        dest='use_amp',
        help='Disable Automatic Mixed Precision'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for super resolution (default: 4)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of worker threads (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize enhancer
        logger.info("Initializing image enhancer...")
        logger.info(f"Scale factor: {args.scale}x")
        
        enhancer = ImageEnhancer(
            device=args.device,
            model_dir=args.model_dir,
            use_amp=args.use_amp,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            scale=args.scale,
            force_fallback=args.force_fallback
        )
        
        # Process directory
        logger.info(f"Processing: {args.source_dir} → {args.dest_dir}")
        logger.info(f"All output files will be saved directly to: {args.dest_dir}")
        
        start_time = datetime.now()
        
        stats = enhancer.process_directory(
            source_dir=args.source_dir,
            dest_dir=args.dest_dir,
            extensions=args.extensions,
            apply_deblur=args.apply_deblur,
            apply_denoise=args.apply_denoise,
            apply_super_res=args.apply_super_res,
            save_intermediate=args.save_intermediate,
            recursive=args.recursive,
            overwrite=args.overwrite,
            filename_suffix=args.suffix,
            parallel=args.parallel,
            scale=args.scale
        )
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        if stats['success'] > 0:
            logger.info(f"Average time per image: {elapsed_time/stats['success']:.2f} seconds")
        
        # Return exit code based on success
        if stats['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()