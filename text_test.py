#!/usr/bin/env python3
"""
Text Support Test - Demonstrates text functionality in the calibrated paper tool.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from py_3d_construct_lib.calibrated_paper import CalibratedPaper


def main():
    """Test text functionality."""
    print("Text Support Test")
    print("=" * 20)
    
    # Create a new calibrated paper
    paper = CalibratedPaper(margin_mm=10.0)
    
    # Add some shapes
    paper.add_rectangle(
        x=20.0, y=20.0,
        width=40.0, height=20.0,
        fill_color="lightblue",
        stroke_color="blue",
        stroke_width=0.2
    )
    
    # Add various text examples
    paper.add_text(
        x=20.0, y=45.0,
        text="Left aligned text",
        font_size=10,
        color="black",
        anchor="left"
    )
    
    paper.add_text(
        x=40.0, y=30.0,
        text="Centered",
        font_size=12,
        color="blue",
        anchor="middle"
    )
    
    paper.add_text(
        x=60.0, y=15.0,
        text="Right aligned",
        font_size=8,
        color="red",
        anchor="right"
    )
    
    paper.add_text(
        x=100.0, y=50.0,
        text="Different Font Size",
        font_size=16,
        color="green",
        anchor="left"
    )
    
    # Add coordinate markers
    paper.add_text(
        x=5.0, y=5.0,
        text="(0,0) here →",
        font_size=6,
        color="gray",
        anchor="right"
    )
    
    paper.print_info()
    
    # Generate output files
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    svg_path = os.path.join(output_dir, "text_test.svg")
    pdf_path = os.path.join(output_dir, "text_test.pdf")
    
    paper.to_svg(svg_path)
    paper.to_pdf(pdf_path)
    
    print(f"\nOutput files:")
    print(f"SVG: {svg_path}")
    print(f"PDF: {pdf_path}")
    
    print(f"\nText Features:")
    print(f"- Multiple font sizes (6pt to 16pt)")
    print(f"- Different colors (black, blue, red, green, gray)")
    print(f"- Text anchoring (left, middle, right)")
    print(f"- Accurate positioning in millimeters")
    print(f"- Bottom-left coordinate system")


if __name__ == "__main__":
    main()
