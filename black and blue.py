import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import textwrap
import random
import re

# --- CONFIGURATION ---
# File paths (ensure these files are in the same folder as the script)
INPUT_TEXT_PATH = 'input.txt'
LINED_PAPER_PATH = 'lined_paper.png'
FONT_PATH = 'PremiumUltra7.ttf'
# OUTPUT_PATH is now generated automatically (e.g., output_1.png, output_2.png)

# Text rendering settings
# These may need tweaking based on your lined_paper.png and font size
FONT_SIZE = 30
LEFT_MARGIN = 85
RIGHT_MARGIN = -20 # New setting for the right margin

TOP_MARGIN = 146
BOTTOM_MARGIN = 80 # How much space to leave at the bottom
LINE_SPACING = 34.7
INK_COLOR = (28, 52, 98, 240) # A slightly transparent dark blue ink
BOLD_COLOR = (0, 0, 0, 255) # Black color for bold text

# Effect settings
NOISE_INTENSITY = 0.01 # How much grain to add (0 to 1)

def add_scanned_noise(image, intensity):
    """Overlays monochrome noise to simulate scanner grain."""
    if intensity <= 0:
        return image

    # Create a random noise array
    noise = np.random.rand(image.height, image.width) * intensity
    
    # Convert image to numpy array and normalize to 0-1
    img_array = np.array(image) / 255.0

    # Add noise to each channel
    for i in range(3): # R, G, B channels
        img_array[:, :, i] = (img_array[:, :, i] + noise).clip(0, 1)

    # Convert back to a displayable image format
    noisy_image = Image.fromarray((img_array * 255).astype(np.uint8))
    
    # Add a final subtle blur to blend the noise
    return noisy_image.filter(ImageFilter.GaussianBlur(radius=0.5))


def parse_text_segments(text):
    """
    Parse text and return a list of segments with their formatting.
    Returns: [(text, is_bold), ...]
    """
    segments = []
    pattern = r'\*\*(.*?)\*\*'
    last_end = 0
    
    for match in re.finditer(pattern, text):
        # Add regular text before the bold text
        if match.start() > last_end:
            segments.append((text[last_end:match.start()], False))
        # Add bold text (without the ** markers)
        segments.append((match.group(1), True))
        last_end = match.end()
    
    # Add any remaining regular text
    if last_end < len(text):
        segments.append((text[last_end:], False))
    
    return segments


def wrap_text_with_formatting(text, chars_per_line):
    """
    Wrap text while preserving bold formatting markers.
    Returns a list of lines with their formatting information.
    """
    # First, parse the segments
    segments = parse_text_segments(text)
    
    lines = []
    current_line = []
    current_length = 0
    
    for segment_text, is_bold in segments:
        words = segment_text.split(' ')
        
        for i, word in enumerate(words):
            word_length = len(word)
            space_length = 1 if i > 0 or current_line else 0
            
            # Check if adding this word would exceed the line length
            if current_length + space_length + word_length > chars_per_line and current_line:
                # Save current line and start a new one
                lines.append(current_line)
                current_line = [(word, is_bold)]
                current_length = word_length
            else:
                # Add space if needed
                if space_length and current_line:
                    # Add space to the last segment if same formatting, or as new segment
                    if current_line[-1][1] == is_bold:
                        current_line[-1] = (current_line[-1][0] + ' ' + word, is_bold)
                    else:
                        current_line.append((' ' + word, is_bold))
                else:
                    current_line.append((word, is_bold))
                current_length += space_length + word_length
    
    # Add the last line if it exists
    if current_line:
        lines.append(current_line)
    
    return lines


def draw_formatted_line(draw, x, y, line_segments, font):
    """
    Draw a line with mixed formatting (regular and bold text).
    line_segments: [(text, is_bold), ...]
    """
    current_x = x
    
    for text, is_bold in line_segments:
        color = BOLD_COLOR if is_bold else INK_COLOR
        draw.text((current_x, y), text, font=font, fill=color)
        # Move x position forward by the width of the text just drawn
        current_x += font.getlength(text)


def main():
    """Main function to generate the handwritten paper image(s)."""
    print("Starting process...")

    # --- 1. Load Assets ---
    try:
        with open(INPUT_TEXT_PATH, 'r', encoding='utf-8') as f:
            text_content = f.read()
        print(f"Successfully read text from '{INPUT_TEXT_PATH}'.")
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_TEXT_PATH}'. Please create it.")
        return

    try:
        # Load a template of the paper to be copied for each new page
        paper_template = Image.open(LINED_PAPER_PATH).convert("RGBA")
        print(f"Successfully loaded background image '{LINED_PAPER_PATH}'.")
    except FileNotFoundError:
        print(f"Error: Could not find '{LINED_PAPER_PATH}'. Please add it to the folder.")
        return

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        print(f"Successfully loaded font '{FONT_PATH}'.")
    except IOError:
        print(f"Error: Could not load font '{FONT_PATH}'. Please add it to the folder.")
        return

    # --- 2. Prepare All Lines for Rendering ---
    paper_width, paper_height = paper_template.size
    avg_char_width = font.getlength("a")
    # Calculate printable width using both left and right margins
    printable_width = paper_width - LEFT_MARGIN - RIGHT_MARGIN
    # Ensure chars_per_line is at least 1 to avoid errors with textwrap
    chars_per_line = max(1, int(printable_width / avg_char_width))

    all_lines = []
    paragraphs = text_content.split('\n')
    for paragraph in paragraphs:
        if not paragraph.strip():
            # Add a blank line to represent a paragraph break's extra spacing
            all_lines.append([])
        else:
            wrapped_lines = wrap_text_with_formatting(paragraph, chars_per_line)
            all_lines.extend(wrapped_lines)
    
    # --- 3. Paginate and Render ---
    page_number = 1
    lines_processed = 0

    while lines_processed < len(all_lines):
        # Create a fresh copy of the paper for the new page
        current_page_image = paper_template.copy()
        draw = ImageDraw.Draw(current_page_image)
        y_position = TOP_MARGIN

        print(f"Writing page {page_number}...")

        # Render lines until the page is full or all lines are drawn
        while y_position < (paper_height - BOTTOM_MARGIN) and lines_processed < len(all_lines):
            line_segments = all_lines[lines_processed]

            # Empty list means blank line for paragraph break
            if line_segments:
                # Add a slight random horizontal jitter for a more natural look
                x_jitter = random.randint(-2, 2)
                draw_formatted_line(
                    draw,
                    LEFT_MARGIN + x_jitter,
                    y_position,
                    line_segments,
                    font
                )

            y_position += LINE_SPACING
            lines_processed += 1

        print(f"Page {page_number} has been written.")

        # --- 4. Apply Effects ---
        # Convert to RGB for effects that don't support alpha channels well
        image_with_text = current_page_image.convert("RGB")

        print("Applying scanned noise effect...")
        final_image = add_scanned_noise(image_with_text, NOISE_INTENSITY)

        # --- 5. Save Output ---
        output_filename = f"output_{page_number}.png"
        final_image.save(output_filename)
        print(f"Page {page_number} saved to '{output_filename}'.")
        
        page_number += 1

    if page_number == 1 and not all_lines:
        print("Warning: No text was found in input.txt. No output file was created.")
    else:
        print("All pages have been generated successfully.")


if __name__ == '__main__':
    main()
