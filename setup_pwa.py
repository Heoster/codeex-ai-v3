#!/usr/bin/env python3
"""
🚀 CodeEx AI - PWA Setup Script
Generates PWA icons, validates manifest, and sets up Progressive Web App features
"""

import os
import json
import sys
from PIL import Image, ImageDraw, ImageFont
import requests
from pathlib import Path

def create_directories():
    """Create necessary directories for PWA assets"""
    directories = [
        'static/images',
        'static/icons',
        'static/screenshots'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def generate_app_icon(size, output_path):
    """Generate a CodeEx AI app icon"""
    # Create a new image with gradient background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Create gradient background
    for i in range(size):
        # Gradient from blue to purple
        r = int(102 + (118 - 102) * i / size)  # 667eea to 764ba2
        g = int(126 + (75 - 126) * i / size)
        b = int(234 + (162 - 234) * i / size)
        
        draw.line([(0, i), (size, i)], fill=(r, g, b, 255))
    
    # Add rounded corners
    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    corner_radius = size // 8
    mask_draw.rounded_rectangle(
        [(0, 0), (size, size)], 
        radius=corner_radius, 
        fill=255
    )
    
    # Apply mask for rounded corners
    img.putalpha(mask)
    
    # Add "C" letter in the center
    try:
        # Try to use a nice font
        font_size = size // 2
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw the "C" letter
    text = "C"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    
    # Add shadow effect
    shadow_offset = max(2, size // 100)
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, 100))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    
    # Save the image
    img.save(output_path, 'PNG', optimize=True)
    print(f"✅ Generated icon: {output_path} ({size}x{size})")

def generate_all_icons():
    """Generate all required PWA icons"""
    icon_sizes = [
        (16, 'favicon-16x16.png'),
        (32, 'favicon-32x32.png'),
        (72, 'icon-72x72.png'),
        (96, 'icon-96x96.png'),
        (128, 'icon-128x128.png'),
        (144, 'icon-144x144.png'),
        (152, 'icon-152x152.png'),
        (180, 'apple-touch-icon.png'),
        (192, 'icon-192x192.png'),
        (384, 'icon-384x384.png'),
        (512, 'icon-512x512.png'),
    ]
    
    # Microsoft tile sizes
    tile_sizes = [
        (70, 'mstile-70x70.png'),
        (144, 'mstile-144x144.png'),
        (150, 'mstile-150x150.png'),
        (310, 'mstile-310x310.png'),
    ]
    
    # Generate regular icons
    for size, filename in icon_sizes:
        output_path = f'static/images/{filename}'
        generate_app_icon(size, output_path)
    
    # Generate Microsoft tiles
    for size, filename in tile_sizes:
        output_path = f'static/images/{filename}'
        generate_app_icon(size, output_path)
    
    # Generate wide tile (310x150)
    generate_wide_tile()
    
    # Generate shortcut icons
    generate_shortcut_icons()

def generate_wide_tile():
    """Generate wide Microsoft tile (310x150)"""
    width, height = 310, 150
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Create gradient background
    for i in range(height):
        r = int(102 + (118 - 102) * i / height)
        g = int(126 + (75 - 126) * i / height)
        b = int(234 + (162 - 234) * i / height)
        draw.line([(0, i), (width, i)], fill=(r, g, b, 255))
    
    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    text = "CodeEx AI"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Add shadow and text
    draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 100))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    
    img.save('static/images/mstile-310x150.png', 'PNG', optimize=True)
    print("✅ Generated wide tile: mstile-310x150.png")

def generate_shortcut_icons():
    """Generate shortcut icons for PWA"""
    shortcuts = [
        ('new-chat', '💬', (34, 139, 34)),  # Forest Green
        ('storage', '💾', (70, 130, 180)),   # Steel Blue
    ]
    
    for name, emoji, color in shortcuts:
        size = 96
        img = Image.new('RGBA', (size, size), color + (255,))
        draw = ImageDraw.Draw(img)
        
        # Add rounded corners
        mask = Image.new('L', (size, size), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle([(0, 0), (size, size)], radius=size//8, fill=255)
        img.putalpha(mask)
        
        # Add emoji/icon
        try:
            font = ImageFont.truetype("seguiemj.ttf", size//2)  # Windows emoji font
        except:
            try:
                font = ImageFont.truetype("arial.ttf", size//3)
            except:
                font = ImageFont.load_default()
        
        text = emoji
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (size - text_width) // 2
        y = (size - text_height) // 2
        
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        
        output_path = f'static/images/shortcut-{name}.png'
        img.save(output_path, 'PNG', optimize=True)
        print(f"✅ Generated shortcut icon: {output_path}")

def generate_screenshots():
    """Generate sample screenshots for PWA manifest"""
    # Mobile screenshot (390x844)
    mobile_img = Image.new('RGB', (390, 844), (10, 10, 15))
    mobile_draw = ImageDraw.Draw(mobile_img)
    
    # Add header
    mobile_draw.rectangle([(0, 0), (390, 80)], fill=(102, 126, 234))
    
    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
        font_text = ImageFont.truetype("arial.ttf", 14)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
    
    mobile_draw.text((20, 30), "CodeEx AI", font=font_title, fill=(255, 255, 255))
    
    # Add chat bubbles
    bubble_y = 120
    messages = [
        ("Hello! How can I help you today?", False),
        ("Can you help me with Python programming?", True),
        ("Of course! I'd be happy to help with Python.", False),
    ]
    
    for message, is_user in messages:
        bubble_width = min(280, len(message) * 8)
        bubble_height = 40
        
        if is_user:
            x = 390 - bubble_width - 20
            color = (102, 126, 234)
        else:
            x = 20
            color = (60, 60, 80)
        
        mobile_draw.rounded_rectangle(
            [(x, bubble_y), (x + bubble_width, bubble_y + bubble_height)],
            radius=20, fill=color
        )
        
        mobile_draw.text((x + 15, bubble_y + 12), message[:30], font=font_text, fill=(255, 255, 255))
        bubble_y += 60
    
    mobile_img.save('static/images/screenshot-mobile.png', 'PNG', optimize=True)
    print("✅ Generated mobile screenshot")
    
    # Desktop screenshot (1920x1080)
    desktop_img = Image.new('RGB', (1920, 1080), (10, 10, 15))
    desktop_draw = ImageDraw.Draw(desktop_img)
    
    # Add header
    desktop_draw.rectangle([(0, 0), (1920, 100)], fill=(102, 126, 234))
    
    try:
        font_large = ImageFont.truetype("arial.ttf", 32)
    except:
        font_large = ImageFont.load_default()
    
    desktop_draw.text((50, 35), "CodeEx AI - Intelligent Assistant", font=font_large, fill=(255, 255, 255))
    
    # Add sidebar
    desktop_draw.rectangle([(0, 100), (350, 1080)], fill=(25, 25, 35))
    desktop_draw.text((30, 150), "Recent Conversations", font=font_title, fill=(200, 200, 200))
    
    # Add main chat area
    desktop_draw.rectangle([(350, 100), (1920, 1080)], fill=(15, 15, 25))
    
    desktop_img.save('static/images/screenshot-desktop.png', 'PNG', optimize=True)
    print("✅ Generated desktop screenshot")

def create_safari_pinned_tab():
    """Create Safari pinned tab SVG icon"""
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="16" height="16" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
        </linearGradient>
    </defs>
    <rect width="16" height="16" rx="2" fill="url(#grad1)"/>
    <text x="8" y="12" font-family="Arial, sans-serif" font-size="10" font-weight="bold" 
          text-anchor="middle" fill="white">C</text>
</svg>'''
    
    with open('static/images/safari-pinned-tab.svg', 'w') as f:
        f.write(svg_content)
    
    print("✅ Generated Safari pinned tab icon")

def validate_manifest():
    """Validate PWA manifest file"""
    try:
        with open('static/manifest.json', 'r') as f:
            manifest = json.load(f)
        
        required_fields = ['name', 'short_name', 'start_url', 'display', 'icons']
        missing_fields = [field for field in required_fields if field not in manifest]
        
        if missing_fields:
            print(f"❌ Manifest validation failed. Missing fields: {missing_fields}")
            return False
        
        # Check icons
        if not manifest['icons'] or len(manifest['icons']) < 2:
            print("❌ Manifest needs at least 2 icons")
            return False
        
        # Check for required icon sizes
        icon_sizes = [icon['sizes'] for icon in manifest['icons']]
        required_sizes = ['192x192', '512x512']
        
        for size in required_sizes:
            if size not in icon_sizes:
                print(f"❌ Missing required icon size: {size}")
                return False
        
        print("✅ PWA manifest validation passed")
        return True
        
    except FileNotFoundError:
        print("❌ Manifest file not found")
        return False
    except json.JSONDecodeError:
        print("❌ Invalid JSON in manifest file")
        return False

def create_og_images():
    """Create Open Graph and Twitter Card images"""
    # Open Graph image (1200x630)
    og_img = Image.new('RGB', (1200, 630), (10, 10, 15))
    og_draw = ImageDraw.Draw(og_img)
    
    # Add gradient background
    for i in range(630):
        r = int(102 + (118 - 102) * i / 630)
        g = int(126 + (75 - 126) * i / 630)
        b = int(234 + (162 - 234) * i / 630)
        og_draw.line([(0, i), (1200, i)], fill=(r, g, b))
    
    try:
        font_large = ImageFont.truetype("arial.ttf", 72)
        font_medium = ImageFont.truetype("arial.ttf", 36)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
    
    # Add title
    title = "CodeEx AI"
    bbox = og_draw.textbbox((0, 0), title, font=font_large)
    title_width = bbox[2] - bbox[0]
    x = (1200 - title_width) // 2
    
    og_draw.text((x + 3, 203), title, font=font_large, fill=(0, 0, 0, 100))  # Shadow
    og_draw.text((x, 200), title, font=font_large, fill=(255, 255, 255))
    
    # Add subtitle
    subtitle = "Your Intelligent AI Assistant"
    bbox = og_draw.textbbox((0, 0), subtitle, font=font_medium)
    subtitle_width = bbox[2] - bbox[0]
    x = (1200 - subtitle_width) // 2
    
    og_draw.text((x + 2, 302), subtitle, font=font_medium, fill=(0, 0, 0, 100))  # Shadow
    og_draw.text((x, 300), subtitle, font=font_medium, fill=(255, 255, 255))
    
    og_img.save('static/images/og-image.png', 'PNG', optimize=True)
    print("✅ Generated Open Graph image")
    
    # Twitter Card image (same as OG for simplicity)
    og_img.save('static/images/twitter-card.png', 'PNG', optimize=True)
    print("✅ Generated Twitter Card image")

def create_badge_icon():
    """Create notification badge icon"""
    size = 72
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Create circular badge
    draw.ellipse([(0, 0), (size, size)], fill=(220, 53, 69, 255))  # Red badge
    
    # Add "C" in center
    try:
        font = ImageFont.truetype("arial.ttf", size//2)
    except:
        font = ImageFont.load_default()
    
    text = "C"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    
    img.save('static/images/badge-72x72.png', 'PNG', optimize=True)
    print("✅ Generated notification badge")

def main():
    """Main setup function"""
    print("🚀 CodeEx AI - PWA Setup")
    print("=" * 40)
    
    try:
        # Check if PIL is available
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("❌ PIL (Pillow) is required for icon generation")
        print("Install with: pip install Pillow")
        sys.exit(1)
    
    steps = [
        ("Creating directories", create_directories),
        ("Generating app icons", generate_all_icons),
        ("Creating screenshots", generate_screenshots),
        ("Creating Safari icon", create_safari_pinned_tab),
        ("Creating social media images", create_og_images),
        ("Creating notification badge", create_badge_icon),
        ("Validating manifest", validate_manifest),
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            result = step_func()
            if result is not False:  # None or True is success
                success_count += 1
        except Exception as e:
            print(f"❌ Failed: {step_name} - {e}")
    
    print("\n" + "=" * 40)
    if success_count == len(steps):
        print("🎉 PWA setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run: python app.py")
        print("2. Open: http://localhost:5000")
        print("3. Test PWA features in Chrome/Edge")
        print("4. Look for 'Install App' button")
        print("\n🔧 PWA Features:")
        print("- ✅ Offline functionality")
        print("- ✅ App-like experience")
        print("- ✅ Push notifications ready")
        print("- ✅ Background sync")
        print("- ✅ Install prompt")
    else:
        print(f"⚠️ Setup completed with {len(steps) - success_count} issues")
        print("Please check the error messages above")
    
    print("\n📱 Test your PWA:")
    print("- Chrome: DevTools > Application > Manifest")
    print("- Lighthouse: Run PWA audit")
    print("- Mobile: Add to Home Screen")

if __name__ == "__main__":
    main()