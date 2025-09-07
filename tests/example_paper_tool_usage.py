#!/usr/bin/env python3
"""
Example usage of the Calibrated Paper tool.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from py_3d_construct_lib.calibrated_paper import CalibratedPaper


def main():
    """Main function demonstrating the calibrated paper tool."""
    print("Calibrated Paper Tool Example")
    print("=" * 30)

    # Create a new calibrated paper with 10mm margin
    paper = CalibratedPaper(margin_mm=10.0)

    # Add the requested rectangles
    print("Adding rectangles...")

    # First rectangle: 10x30 mm at position (20, 20)
    # Note: Using CAD coordinate system - origin at bottom-left, Y+ goes up
    paper.add_rectangle(
        x=20.0,
        y=20.0,  # 20mm from left, 20mm from bottom
        width=10.0,
        height=30.0,
        fill_color="lightblue",
        stroke_color="blue",
        stroke_width=0.2,
    )

    # Second rectangle: 7x40 mm at position (50, 40)
    paper.add_rectangle(
        x=50.0,
        y=40.0,  # 50mm from left, 40mm from bottom
        width=7.0,
        height=40.0,
        fill_color="lightgreen",
        stroke_color="green",
        stroke_width=0.2,
    )

    # Print information
    paper.print_info()

    # Generate output files
    print("\nGenerating output files...")

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    svg_path = os.path.join(output_dir, "rectangles.svg")
    pdf_path = os.path.join(output_dir, "rectangles.pdf")

    paper.to_svg(svg_path)
    paper.to_pdf(pdf_path)

    print(f"SVG file: {svg_path}")
    print(f"PDF file: {pdf_path}")

    print("\nPrinting Instructions:")
    print("1. Open the PDF file in a PDF viewer")
    print("2. Print with 'Actual Size' or '100% scale' (no 'Fit to Page')")
    print("3. Use a ruler to measure the rectangles:")
    print("   - First rectangle should be 10.0mm × 30.0mm")
    print("   - Second rectangle should be 7.0mm × 40.0mm")
    print("4. If measurements are off, check printer settings")

    print("\nTool Pipeline Summary:")
    print("- Geometry defined in millimeters")
    print("- CAD/3D printing coordinate system: origin at bottom-left, Y+ goes up")
    print("- SVG generation with accurate mm positioning")
    print("- PDF generation using ReportLab with proper scaling")
    print("- Designed for A4 paper (210mm × 297mm)")
    print("- Coordinate system: (0,0) at bottom-left of printable area")


if __name__ == "__main__":
    main()
