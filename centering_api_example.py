#!/usr/bin/env python3
"""
Centering API Example - Shows how to use the centering feature.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from py_3d_construct_lib.calibrated_paper import CalibratedPaper


def main():
    """Example of using the centering feature."""
    print("Centering API Example")
    print("=" * 25)
    
    # Create a document with some objects
    paper = CalibratedPaper(margin_mm=15.0)
    
    # Add objects positioned in one corner
    paper.add_rectangle(x=10.0, y=10.0, width=20.0, height=15.0, fill_color="lightblue")
    paper.add_rectangle(x=35.0, y=20.0, width=15.0, height=10.0, fill_color="lightgreen")
    paper.add_text(x=25.0, y=30.0, text="Example", font_size=12, anchor="middle")
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Method 1: Save individual files with centering
    print("Method 1: Individual file generation")
    paper.to_svg(os.path.join(output_dir, "api_example.svg"), center_objects=True)
    paper.to_pdf(os.path.join(output_dir, "api_example.pdf"), center_objects=True)
    
    # Method 2: Save both files at once with centering
    print("Method 2: Save both files at once")
    paper.save_both(os.path.join(output_dir, "api_example_both"), center_objects=True)
    
    # Method 3: Save without centering (default behavior)
    print("Method 3: Default positioning (no centering)")
    paper.save_both(os.path.join(output_dir, "api_example_default"), center_objects=False)
    
    print("\nAPI Usage Summary:")
    print("- to_svg(path, center_objects=True)")
    print("- to_pdf(path, center_objects=True)")
    print("- save_both(base_path, center_objects=True)")
    print("- Centering calculates bounding box and centers all objects")
    print("- Original positioning is preserved when center_objects=False")


if __name__ == "__main__":
    main()
