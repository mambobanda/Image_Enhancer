#!/usr/bin/env python
"""
Neural Style Transfer - Command Line Tool
Usage: python style_transfer_cli.py --content content.jpg --style style.jpg --output output.jpg
"""

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import argparse
import os
import time
from pathlib import Path


class StyleTransfer:
    """Neural Style Transfer implementation"""
    
    def __init__(self, device="auto", image_size=512, steps=1000, 
                 content_weight=1, style_weight=1e6, lr=0.003):
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.image_size = image_size
        self.steps = steps
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.lr = lr
        
        # Load VGG19
        print(f"Loading VGG19 on {self.device}...")
        self.vgg = models.vgg19(pretrained=True).features.to(self.device).eval()
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        
        # Layer definitions
        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        
        # Default style weights
        self.style_weights = {
            'conv1_1': 1.0,
            'conv2_1': 0.75,
            'conv3_1': 0.2,
            'conv4_1': 0.2,
            'conv5_1': 0.2
        }
        
        print("✅ StyleTransfer initialized")
    
    def load_image(self, img_path, shape=None):
        """Load and preprocess image"""
        image = Image.open(img_path).convert('RGB')
        
        # Resize
        if max(image.size) > self.image_size:
            size = self.image_size
        else:
            size = max(image.size)
        
        if shape is not None:
            size = shape
        
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        image = transform(image)[:3, :, :].unsqueeze(0)
        return image.to(self.device)
    
    def save_image(self, tensor, path):
        """Save tensor as image"""
        image = tensor.to("cpu").clone().detach()
        image = image.squeeze(0)
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image = image.clip(0, 1)
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_image(image, path)
    
    def get_features(self, image, layers=None):
        """Extract features from VGG19"""
        if layers is None:
            layers = {str(i): layer for i, layer in enumerate([
                'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 
                'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2',
                'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'
            ])}
        
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features
    
    def gram_matrix(self, tensor):
        """Calculate Gram matrix"""
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        return torch.mm(tensor, tensor.t())
    
    def content_loss(self, target_features, content_features):
        """Calculate content loss"""
        return torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    
    def style_loss(self, target_features, style_features):
        """Calculate style loss"""
        loss = 0
        for layer in self.style_weights:
            target_feature = target_features[layer]
            target_gram = self.gram_matrix(target_feature)
            style_gram = self.gram_matrix(style_features[layer])
            layer_loss = torch.mean((target_gram - style_gram) ** 2)
            loss += self.style_weights[layer] * layer_loss
        return loss
    
    def transfer(self, content_path, style_path, output_path, 
                 save_intermediate=False, verbose=True):
        """Perform style transfer"""
        
        print(f"\n📸 Content: {content_path}")
        print(f"🎨 Style: {style_path}")
        print(f"💾 Output: {output_path}")
        print(f"⚙️  Steps: {self.steps}, Size: {self.image_size}")
        print("-" * 50)
        
        # Load images
        print("Loading images...")
        content_img = self.load_image(content_path)
        style_img = self.load_image(style_path, shape=content_img.shape[-2:])
        
        # Get features
        print("Extracting features...")
        content_features = self.get_features(content_img)
        style_features = self.get_features(style_img)
        
        # Initialize target
        target = content_img.clone().requires_grad_(True).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam([target], lr=self.lr)
        
        # Processing
        print(f"Starting style transfer ({self.steps} steps)...")
        start_time = time.time()
        
        for step in range(self.steps):
            target_features = self.get_features(target)
            
            # Calculate losses
            c_loss = self.content_loss(target_features, content_features)
            s_loss = self.style_loss(target_features, style_features)
            total_loss = self.content_weight * c_loss + self.style_weight * s_loss
            
            # Update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Progress
            if (step + 1) % 100 == 0 or step == 0:
                elapsed = time.time() - start_time
                print(f"Step {step+1}/{self.steps} | "
                      f"Content: {c_loss.item():.2f} | "
                      f"Style: {s_loss.item():.2f} | "
                      f"Time: {elapsed:.1f}s")
                
                # Save intermediate
                if save_intermediate:
                    inter_path = f"{output_path}_{step+1}_intermediate.jpg"
                    self.save_image(target, inter_path)
                    print(f"  💾 Saved: {inter_path}")
        
        # Save final
        print(f"\n💾 Saving final result...")
        self.save_image(target, output_path)
        
        total_time = time.time() - start_time
        print(f"✅ Done! Time: {total_time:.1f}s")
        print(f"📁 Output: {output_path}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Neural Style Transfer - Command Line Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Basic usage
  python style_transfer_cli.py --content photo.jpg --style painting.jpg --output result.jpg
  
  # With custom parameters
  python style_transfer_cli.py -c photo.jpg -s painting.jpg -o result.jpg --steps 2000 --size 400
  
  # High quality with intermediate saves
  python style_transfer_cli.py -c photo.jpg -s painting.jpg -o result.jpg --steps 3000 --save-intermediate
  
  # Low quality for quick preview
  python style_transfer_cli.py -c photo.jpg -s painting.jpg -o result.jpg --steps 300 --size 256
        """
    )
    
    # Required arguments
    parser.add_argument('-c', '--content', required=True,
                       help='Path to content image')
    parser.add_argument('-s', '--style', required=True,
                       help='Path to style image')
    parser.add_argument('-o', '--output', default='styled_output.jpg',
                       help='Path to output image (default: styled_output.jpg)')
    
    # Optional arguments
    parser.add_argument('--size', type=int, default=512,
                       help='Maximum image size (default: 512)')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Number of optimization steps (default: 1000)')
    parser.add_argument('--content-weight', type=float, default=1,
                       help='Content weight (default: 1)')
    parser.add_argument('--style-weight', type=float, default=1e6,
                       help='Style weight (default: 1000000)')
    parser.add_argument('--lr', type=float, default=0.003,
                       help='Learning rate (default: 0.003)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate results')
    parser.add_argument('--no-verbose', action='store_true',
                       help='Disable verbose output')
    parser.add_argument('--style-weights', type=float, nargs=5,
                       default=[1.0, 0.75, 0.2, 0.2, 0.2],
                       metavar=('conv1', 'conv2', 'conv3', 'conv4', 'conv5'),
                       help='Style weights for conv1-conv5 layers')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.content):
        print(f"❌ Error: Content file not found: {args.content}")
        return 1
    
    if not os.path.exists(args.style):
        print(f"❌ Error: Style file not found: {args.style}")
        return 1
    
    try:
        # Initialize style transfer
        styler = StyleTransfer(
            device=args.device,
            image_size=args.size,
            steps=args.steps,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            lr=args.lr
        )
        
        # Custom style weights
        styler.style_weights = {
            'conv1_1': args.style_weights[0],
            'conv2_1': args.style_weights[1],
            'conv3_1': args.style_weights[2],
            'conv4_1': args.style_weights[3],
            'conv5_1': args.style_weights[4]
        }
        
        # Run style transfer
        output = styler.transfer(
            content_path=args.content,
            style_path=args.style,
            output_path=args.output,
            save_intermediate=args.save_intermediate,
            verbose=not args.no_verbose
        )
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())