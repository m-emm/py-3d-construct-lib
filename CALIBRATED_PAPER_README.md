# Calibrated Paper Tool

A Python tool for creating accurately positioned geometry in millimeters and converting it to SVG and PDF formats for precise printing on A4 paper.

## Overview

The Calibrated Paper tool provides a pipeline for:
1. **Geometry Definition**: Define shapes (starting with rectangles) in millimeters
2. **SVG Generation**: Convert to SVG with accurate millimeter positioning
3. **PDF Generation**: Convert to PDF with proper scaling for printing
4. **Calibrated Printing**: Print with 0.1mm accuracy on A4 paper

## Key Features

- **Millimeter-based coordinates**: All dimensions and positions in mm
- **A4 paper support**: 210mm × 297mm with configurable margins
- **Accurate scaling**: Proper DPI handling for printing
- **Multiple output formats**: SVG and PDF generation
- **Print calibration**: Optimized for European A4 printers

## Installation

```bash
# Install required dependencies
pip install svgwrite reportlab weasyprint
```

## Quick Start

```python
from py_3d_construct_lib.calibrated_paper import CalibratedPaper

# Create a new calibrated paper with 10mm margin
paper = CalibratedPaper(margin_mm=10.0)

# Add rectangles (dimensions in mm)
paper.add_rectangle(
    x=20.0, y=20.0,        # Position: 20mm from left, 20mm from bottom
    width=10.0, height=30.0, # Size: 10mm × 30mm
    fill_color="lightblue",
    stroke_color="blue",
    stroke_width=0.2
)

paper.add_rectangle(
    x=50.0, y=40.0,        # Position: 50mm from left, 40mm from bottom
    width=7.0, height=40.0,  # Size: 7mm × 40mm
    fill_color="lightgreen",
    stroke_color="green",
    stroke_width=0.2
)

# Generate output files
paper.to_svg("output.svg")
paper.to_pdf("output.pdf")
```

## Coordinate System

- Origin (0,0) is at the bottom-left of the printable area
- X-axis increases to the right
- Y-axis increases upward
- All measurements in millimeters
- Printable area excludes the specified margin
- **Compatible with CAD and 3D printing conventions**

## Pipeline Tools

### 1. Geometry Definition
- **Rectangle**: Define rectangles with position, size, and styling
- **Colors**: Support for standard colors and hex values
- **Stroke width**: Configurable line thickness in mm

### 2. SVG Generation
- Uses `svgwrite` library
- Accurate millimeter positioning
- Proper viewBox scaling
- CSS color support

### 3. PDF Generation
- Uses `reportlab` library
- A4 page size (210mm × 297mm)
- Accurate coordinate transformation
- Print-ready output

## Printing for Maximum Accuracy

1. **Open PDF**: Use a standard PDF viewer (Preview, Adobe Reader, etc.)
2. **Print Settings**: 
   - Select "Actual Size" or "100% scale"
   - **Never** use "Fit to Page" or "Scale to Fit"
   - Use "Quality" or "Best" print settings
3. **Verification**: 
   - Measure printed rectangles with a ruler
   - First rectangle should be exactly 10.0mm × 30.0mm
   - Second rectangle should be exactly 7.0mm × 40.0mm

## Troubleshooting

### Measurements are off by a constant factor
- Check print settings (ensure "Actual Size" is selected)
- Verify printer is not applying automatic scaling
- Check paper size settings (should be A4)

### Measurements are off by small amounts (< 1mm)
- Printer calibration may be needed
- Different paper types can affect accuracy
- Check for paper feed issues

### Colors don't match
- Monitor vs. print color differences are normal
- Use CMYK colors for better print matching
- Consider printer color profiles

## Example Output

The tool creates two test rectangles with CAD-style coordinate system:
- **Rectangle 1**: 10mm × 30mm at position (20, 20) - 20mm from left, 20mm from bottom
- **Rectangle 2**: 7mm × 40mm at position (50, 40) - 50mm from left, 40mm from bottom

## API Reference

### CalibratedPaper Class

```python
class CalibratedPaper:
    def __init__(self, margin_mm: float = 10.0)
    def add_rectangle(self, x, y, width, height, fill_color="none", stroke_color="black", stroke_width=0.1)
    def to_svg(self, output_path: str)
    def to_pdf(self, output_path: str)
    def clear(self)
    def print_info(self)
```

### Rectangle Class

```python
@dataclass
class Rectangle:
    x: float          # X position in mm
    y: float          # Y position in mm  
    width: float      # Width in mm
    height: float     # Height in mm
    fill_color: str   # Fill color
    stroke_color: str # Stroke color
    stroke_width: float # Stroke width in mm
```

## Future Enhancements

- **Additional shapes**: Circles, ellipses, polygons
- **Text support**: Add text with specific fonts and sizes
- **Grid system**: Optional grid overlay for alignment
- **Multiple pages**: Support for multi-page documents
- **Templates**: Common layout templates
- **Measurement tools**: Built-in rulers and dimension lines

## Testing

Run the example to verify the tool works correctly:

```bash
python example_usage.py
```

This will create test files in the `output/` directory that you can print and measure to verify accuracy.
