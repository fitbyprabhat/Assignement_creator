#!/usr/bin/env python3
"""
Scanned Document Simulator
Converts images to realistic scanned PDFs with various artifacts
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io

class ScanSimulator:
    def __init__(self, input_folder="images", output_folder="scanned_pdfs"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
    def add_noise(self, img, intensity=0.02):
        """Add random noise to simulate scanner artifacts"""
        img_array = np.array(img)
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    def add_blur(self, img):
        """Add slight blur to simulate scanner motion"""
        blur_amount = random.uniform(0.1, 0.3)
        return img.filter(ImageFilter.GaussianBlur(blur_amount))
    
    def adjust_brightness_contrast(self, img):
        """Adjust brightness and contrast randomly"""
        # Brightness
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(random.uniform(0.9, 1.1))
        
        # Contrast
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(random.uniform(0.95, 1.15))
        
        return img
    
    def add_rotation(self, img):
        """Add slight rotation to simulate imperfect scanning"""
        angle = random.uniform(-1.5, 1.5)
        # Use white background and add padding to avoid jagged edges
        # First expand the canvas
        width, height = img.size
        padding = int(max(width, height) * 0.05)  # 5% padding
        
        # Create new image with padding
        new_size = (width + 2 * padding, height + 2 * padding)
        padded = Image.new('RGB', new_size, 'white')
        padded.paste(img, (padding, padding))
        
        # Rotate with white fill
        rotated = padded.rotate(angle, expand=False, fillcolor='white', resample=Image.BICUBIC)
        
        # Crop back to original size (centered)
        left = padding
        top = padding
        right = padding + width
        bottom = padding + height
        
        return rotated.crop((left, top, right, bottom))
    
    def add_shadow(self, img):
        """Add realistic book/document shadow effect"""
        width, height = img.size
        img_array = np.array(img.convert('RGB'))
        
        # Create shadow overlay
        shadow = np.ones((height, width), dtype=np.float32)
        
        # Add center shadow (book binding effect) - more subtle
        center_shadow = random.choice([True, False])
        if center_shadow:
            center_x = width // 2
            shadow_width = random.randint(40, 100)
            shadow_darkness = random.uniform(0.88, 0.95)  # Much lighter
            
            for x in range(max(0, center_x - shadow_width), min(width, center_x + shadow_width)):
                distance = abs(x - center_x)
                # Smoother gradient with exponential falloff
                normalized_dist = distance / shadow_width
                factor = 1 - (1 - shadow_darkness) * np.exp(-3 * normalized_dist)
                shadow[:, x] *= factor
        
        # Add random vertical fold/crease shadows (like paper folds)
        num_vertical_folds = random.randint(0, 2)  # 0-2 random vertical shadows
        for _ in range(num_vertical_folds):
            fold_x = random.randint(width // 4, 3 * width // 4)
            fold_width = random.randint(30, 70)
            fold_darkness = random.uniform(0.90, 0.96)
            
            for x in range(max(0, fold_x - fold_width), min(width, fold_x + fold_width)):
                distance = abs(x - fold_x)
                normalized_dist = distance / fold_width
                factor = 1 - (1 - fold_darkness) * np.exp(-3 * normalized_dist)
                shadow[:, x] *= factor
        
        # Add random horizontal fold/crease shadows
        num_horizontal_folds = random.randint(0, 2)  # 0-2 random horizontal shadows
        for _ in range(num_horizontal_folds):
            fold_y = random.randint(height // 4, 3 * height // 4)
            fold_height = random.randint(30, 70)
            fold_darkness = random.uniform(0.90, 0.96)
            
            for y in range(max(0, fold_y - fold_height), min(height, fold_y + fold_height)):
                distance = abs(y - fold_y)
                normalized_dist = distance / fold_height
                factor = 1 - (1 - fold_darkness) * np.exp(-3 * normalized_dist)
                shadow[y, :] *= factor
        
        # Add diagonal crease shadows (occasional)
        if random.random() < 0.3:  # 30% chance
            diagonal_direction = random.choice(['tl_br', 'tr_bl'])  # top-left to bottom-right or vice versa
            crease_width = random.randint(40, 80)
            crease_darkness = random.uniform(0.91, 0.96)
            
            if diagonal_direction == 'tl_br':
                # Diagonal from top-left to bottom-right
                diagonal_offset = random.randint(-width // 3, width // 3)
                for y in range(height):
                    center_x = int((y / height) * width + diagonal_offset)
                    for x in range(max(0, center_x - crease_width), min(width, center_x + crease_width)):
                        distance = abs(x - center_x)
                        normalized_dist = distance / crease_width
                        factor = 1 - (1 - crease_darkness) * np.exp(-3 * normalized_dist)
                        shadow[y, x] *= factor
            else:
                # Diagonal from top-right to bottom-left
                diagonal_offset = random.randint(-width // 3, width // 3)
                for y in range(height):
                    center_x = int(width - (y / height) * width + diagonal_offset)
                    for x in range(max(0, center_x - crease_width), min(width, center_x + crease_width)):
                        distance = abs(x - center_x)
                        normalized_dist = distance / crease_width
                        factor = 1 - (1 - crease_darkness) * np.exp(-3 * normalized_dist)
                        shadow[y, x] *= factor
        
        # Add corner shadows - more subtle
        corner_intensity = random.uniform(0.85, 0.93)
        for corner in random.sample(['tl', 'tr', 'bl', 'br'], random.randint(1, 3)):
            shadow_size = random.randint(height // 8, height // 5)
            
            if corner == 'tl':  # Top-left
                for i in range(shadow_size):
                    for j in range(shadow_size):
                        dist = np.sqrt(i**2 + j**2) / shadow_size
                        shadow[i, j] *= corner_intensity + (1 - corner_intensity) * (dist ** 0.7)
            elif corner == 'tr':  # Top-right
                for i in range(shadow_size):
                    for j in range(shadow_size):
                        dist = np.sqrt(i**2 + j**2) / shadow_size
                        shadow[i, width - 1 - j] *= corner_intensity + (1 - corner_intensity) * (dist ** 0.7)
            elif corner == 'bl':  # Bottom-left
                for i in range(shadow_size):
                    for j in range(shadow_size):
                        dist = np.sqrt(i**2 + j**2) / shadow_size
                        shadow[height - 1 - i, j] *= corner_intensity + (1 - corner_intensity) * (dist ** 0.7)
            elif corner == 'br':  # Bottom-right
                for i in range(shadow_size):
                    for j in range(shadow_size):
                        dist = np.sqrt(i**2 + j**2) / shadow_size
                        shadow[height - 1 - i, width - 1 - j] *= corner_intensity + (1 - corner_intensity) * (dist ** 0.7)
        
        # Add edge shadows (random sides) - very subtle
        edges = random.sample(['top', 'bottom', 'left', 'right'], random.randint(1, 2))
        edge_intensity = random.uniform(0.92, 0.97)
        edge_size = random.randint(10, 25)
        
        for edge in edges:
            if edge == 'top':
                for i in range(min(edge_size, height)):
                    factor = edge_intensity + (1 - edge_intensity) * (i / edge_size) ** 0.5
                    shadow[i, :] *= factor
            elif edge == 'bottom':
                for i in range(min(edge_size, height)):
                    factor = edge_intensity + (1 - edge_intensity) * (i / edge_size) ** 0.5
                    shadow[height - 1 - i, :] *= factor
            elif edge == 'left':
                for j in range(min(edge_size, width)):
                    factor = edge_intensity + (1 - edge_intensity) * (j / edge_size) ** 0.5
                    shadow[:, j] *= factor
            elif edge == 'right':
                for j in range(min(edge_size, width)):
                    factor = edge_intensity + (1 - edge_intensity) * (j / edge_size) ** 0.5
                    shadow[:, width - 1 - j] *= factor
        
        # Apply shadow to image
        shadow_3d = np.stack([shadow] * 3, axis=-1)
        shadowed = (img_array * shadow_3d).astype(np.uint8)
        
        result = Image.fromarray(shadowed)
        
        # Add slight blur to make shadows more realistic
        return result.filter(ImageFilter.GaussianBlur(0.3))
    
    def add_texture(self, img):
        """Add paper texture"""
        width, height = img.size
        texture = np.random.randint(250, 256, (height, width, 3), dtype=np.uint8)
        texture_img = Image.fromarray(texture)
        
        return Image.blend(img.convert('RGB'), texture_img, alpha=0.03)
    
    def simulate_scan(self, img):
        """Apply all scanning effects"""
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply effects - reduced blur frequency
        effects = [
            lambda x: self.add_rotation(x),
            lambda x: self.add_noise(x, intensity=random.uniform(0.005, 0.015)),
            lambda x: self.adjust_brightness_contrast(x),
            lambda x: self.add_shadow(x),
            lambda x: self.add_texture(x)
        ]
        
        # Always apply these core effects
        for effect in effects:
            img = effect(img)
        
        # Only occasionally add blur (30% chance)
        if random.random() < 0.3:
            img = self.add_blur(img)
        
        return img
    
    def create_pdf(self, images, output_path, page_size=A4):
        """Create PDF from list of images"""
        c = canvas.Canvas(str(output_path), pagesize=page_size)
        page_width, page_height = page_size
        
        for img in images:
            # Convert PIL Image to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=85)
            img_buffer.seek(0)
            
            # Get image dimensions
            img_width, img_height = img.size
            
            # Calculate scaling to fit page while maintaining aspect ratio
            width_ratio = page_width / img_width
            height_ratio = page_height / img_height
            ratio = min(width_ratio, height_ratio) * 0.95  # 95% to add margins
            
            new_width = img_width * ratio
            new_height = img_height * ratio
            
            # Center image on page
            x = (page_width - new_width) / 2
            y = (page_height - new_height) / 2
            
            # Draw image
            c.drawImage(ImageReader(img_buffer), x, y, new_width, new_height)
            c.showPage()
        
        c.save()
        print(f"✓ Saved PDF: {output_path}")
    
    def natural_sort_key(self, filename):
        """Natural sorting key function for proper numeric ordering"""
        import re
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split('([0-9]+)', str(filename))]
    
    def process_folder(self, combine_into_single_pdf=False):
        """Process all images in the input folder"""
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        
        # Find all images
        image_files = [
            f for f in self.input_folder.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No images found in {self.input_folder}")
            return
        
        # Sort images by name with natural sorting (handles numbers correctly)
        image_files = sorted(image_files, key=lambda x: self.natural_sort_key(x.name))
        
        print(f"Found {len(image_files)} images")
        print(f"Processing in order: {', '.join([f.name for f in image_files])}")
        
        if combine_into_single_pdf:
            # Process all images and combine into one PDF
            scanned_images = []
            for img_path in image_files:
                print(f"Processing: {img_path.name}")
                img = Image.open(img_path)
                scanned_img = self.simulate_scan(img)
                scanned_images.append(scanned_img)
            
            output_path = self.output_folder / "combined_scanned.pdf"
            self.create_pdf(scanned_images, output_path)
            print(f"\n✓ Created combined PDF with {len(scanned_images)} pages")
        else:
            # Create separate PDF for each image
            for img_path in image_files:
                print(f"Processing: {img_path.name}")
                img = Image.open(img_path)
                scanned_img = self.simulate_scan(img)
                
                # Create PDF with same name as image
                output_path = self.output_folder / f"{img_path.stem}_scanned.pdf"
                self.create_pdf([scanned_img], output_path)
            
            print(f"\n✓ Created {len(image_files)} scanned PDFs")

def main():
    print("=" * 60)
    print("Scanned Document Simulator")
    print("=" * 60)
    
    # Configuration
    INPUT_FOLDER = "."  # Current folder (where the script is)
    OUTPUT_FOLDER = "scanned_pdfs"
    COMBINE_PDF = True  # Set to True to combine all images into one PDF
    
    # Create simulator
    simulator = ScanSimulator(INPUT_FOLDER, OUTPUT_FOLDER)
    
    # Process images
    simulator.process_folder(combine_into_single_pdf=COMBINE_PDF)
    
    print("\n✓ All done!")

if __name__ == "__main__":
    main()
