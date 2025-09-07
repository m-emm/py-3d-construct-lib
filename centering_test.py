#!/usr/bin/env python3
"""
Centering Test - Demonstrates the centering functionality.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from py_3d_construct_lib.calibrated_paper import CalibratedPaper


def main():
    """Test centering functionality."""
    print("Centering Test")
    print("=" * 20)
    
    # Create a new calibrated paper
    paper = CalibratedPaper(margin_mm=10.0)
    
    # Add some objects in a corner to demonstrate centering
    paper.add_rectangle(
        x=10.0, y=10.0,
        width=30.0, height=20.0,
        fill_color="lightblue",
        stroke_color="blue",
        stroke_width=0.2
    )
    
    paper.add_rectangle(
        x=50.0, y=15.0,
        width=20.0, height=15.0,
        fill_color="lightgreen",
        stroke_color="green",
        stroke_width=0.2
    )
    
    paper.add_text(
        x=25.0, y=35.0,
        text="Test Objects",
        font_size=14,
        color="black",
        anchor="middle"
    )
    
    paper.add_text(
        x=60.0, y=5.0,
        text="Bottom Right",
        font_size=10,
        color="red",
        anchor="right"
    )
    
    paper.print_info()
    
    # Show bounding box calculation
    min_x, min_y, max_x, max_y = paper._calculate_bounding_box()
    print(f"\nBounding box: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})")
    print(f"Object dimensions: {max_x - min_x:.1f} × {max_y - min_y:.1f} mm")
    
    offset_x, offset_y = paper._get_centering_offset()
    print(f"Centering offset: ({offset_x:.1f}, {offset_y:.1f}) mm")
    
    # Generate output files
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save without centering
    print(f"\nSaving without centering...")
    paper.save_both(os.path.join(output_dir, "test_no_center"), center_objects=False)
    
    # Save with centering
    print(f"\nSaving with centering...")
    paper.save_both(os.path.join(output_dir, "test_centered"), center_objects=True)
    
    print(f"\nComparison:")
    print(f"- test_no_center.*: Objects positioned as specified")
    print(f"- test_centered.*: Objects centered on the page")
    print(f"- Both maintain accurate dimensions and proportions")


if __name__ == "__main__":
    main()
