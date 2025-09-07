#!/usr/bin/env python3
"""
Coordinate System Test - Shows the difference between old and new coordinate systems.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from py_3d_construct_lib.calibrated_paper import CalibratedPaper


def main():
    """Demonstrate the CAD-style coordinate system."""
    print("CAD/3D Printing Coordinate System Test")
    print("=" * 40)
    
    # Create a new calibrated paper with smaller margin for better visibility
    paper = CalibratedPaper(margin_mm=5.0)
    
    print(f"Printable area: {paper.printable_width}mm × {paper.printable_height}mm")
    print("Coordinate system: Origin (0,0) at bottom-left, Y+ goes up")
    print()
    
    # Create a coordinate system demonstration
    # Add corner markers to show the coordinate system
    
    # Bottom-left corner (origin)
    paper.add_rectangle(
        x=0.0, y=0.0,
        width=5.0, height=5.0,
        fill_color="red",
        stroke_color="darkred",
        stroke_width=0.3
    )
    
    # Bottom-right corner
    paper.add_rectangle(
        x=paper.printable_width - 5.0, y=0.0,
        width=5.0, height=5.0,
        fill_color="blue",
        stroke_color="darkblue",
        stroke_width=0.3
    )
    
    # Top-left corner
    paper.add_rectangle(
        x=0.0, y=paper.printable_height - 5.0,
        width=5.0, height=5.0,
        fill_color="green",
        stroke_color="darkgreen",
        stroke_width=0.3
    )
    
    # Top-right corner
    paper.add_rectangle(
        x=paper.printable_width - 5.0, y=paper.printable_height - 5.0,
        width=5.0, height=5.0,
        fill_color="yellow",
        stroke_color="orange",
        stroke_width=0.3
    )
    
    # Add a small test rectangle at a known position
    paper.add_rectangle(
        x=20.0, y=20.0,  # 20mm from left, 20mm from bottom
        width=10.0, height=30.0,
        fill_color="lightblue",
        stroke_color="blue",
        stroke_width=0.2
    )
    
    # Add text indicators (as rectangles for now)
    # X-axis indicator (horizontal line)
    paper.add_rectangle(
        x=10.0, y=10.0,
        width=30.0, height=1.0,
        fill_color="black",
        stroke_color="black",
        stroke_width=0.1
    )
    
    # Y-axis indicator (vertical line)
    paper.add_rectangle(
        x=10.0, y=10.0,
        width=1.0, height=30.0,
        fill_color="black",
        stroke_color="black",
        stroke_width=0.1
    )
    
    paper.print_info()
    
    # Generate output files
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    svg_path = os.path.join(output_dir, "coordinate_system_test.svg")
    pdf_path = os.path.join(output_dir, "coordinate_system_test.pdf")
    
    paper.to_svg(svg_path)
    paper.to_pdf(pdf_path)
    
    print(f"\nOutput files:")
    print(f"SVG: {svg_path}")
    print(f"PDF: {pdf_path}")
    
    print(f"\nCoordinate System Legend:")
    print(f"- RED square: Origin (0,0) - bottom-left")
    print(f"- BLUE square: Bottom-right")
    print(f"- GREEN square: Top-left")
    print(f"- YELLOW square: Top-right")
    print(f"- Light blue rectangle: Test shape at (20,20)")
    print(f"- Black lines: X and Y axis indicators")
    
    print(f"\nThis matches CAD and 3D printing conventions!")


if __name__ == "__main__":
    main()
