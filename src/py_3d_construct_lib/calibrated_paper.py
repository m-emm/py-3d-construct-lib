"""
Calibrated Paper Tool - Accurate millimeter-based geometry rendering for printing.

This module provides tools to create geometry in millimeters and convert it to
SVG and PDF formats with accurate positioning for printing on A4 paper.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import svgwrite
from reportlab.lib.colors import Color, black, blue, green, lightgrey, red
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


@dataclass
class Rectangle:
    """A rectangle defined in millimeters."""

    x: float  # X position in mm
    y: float  # Y position in mm
    width: float  # Width in mm
    height: float  # Height in mm
    fill_color: str = "none"
    stroke_color: str = "black"
    stroke_width: float = 0.1  # mm


@dataclass
class Text:
    """A text element defined in millimeters."""

    x: float  # X position in mm
    y: float  # Y position in mm
    text: str  # Text content
    font_size: float = 12.0  # Font size in points
    font_family: str = "Arial"  # Font family
    color: str = "black"  # Text color
    anchor: str = "left"  # Text anchor: left, middle, right


class CalibratedPaper:
    """
    A tool for creating accurately positioned geometry on A4 paper.

    Coordinates are specified in millimeters with (0,0) at bottom-left.
    The coordinate system matches CAD and 3D printing conventions:
    - X+ goes to the right
    - Y+ goes up
    - Origin (0,0) is at bottom-left of printable area
    """

    # A4 dimensions in millimeters
    A4_WIDTH_MM = 210.0
    A4_HEIGHT_MM = 297.0

    # Standard margins (can be adjusted)
    DEFAULT_MARGIN_MM = 10.0

    def __init__(self, margin_mm: float = DEFAULT_MARGIN_MM):
        """
        Initialize the calibrated paper tool.

        Args:
            margin_mm: Margin around the printable area in millimeters
        """
        self.margin_mm = margin_mm
        self.printable_width = self.A4_WIDTH_MM - 2 * margin_mm
        self.printable_height = self.A4_HEIGHT_MM - 2 * margin_mm
        self.rectangles: List[Rectangle] = []
        self.texts: List[Text] = []

    def add_rectangle(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill_color: str = "none",
        stroke_color: str = "black",
        stroke_width: float = 0.1,
    ) -> None:
        """
        Add a rectangle to the paper.

        Args:
            x: X position in mm (from left edge of printable area)
            y: Y position in mm (from bottom edge of printable area)
            width: Width in mm
            height: Height in mm
            fill_color: Fill color (CSS color string)
            stroke_color: Stroke color (CSS color string)
            stroke_width: Stroke width in mm
        """
        rect = Rectangle(x, y, width, height, fill_color, stroke_color, stroke_width)
        self.rectangles.append(rect)

    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        font_size: float = 12.0,
        font_family: str = "Arial",
        color: str = "black",
        anchor: str = "left",
    ) -> None:
        """
        Add text to the paper.

        Args:
            x: X position in mm (from left edge of printable area)
            y: Y position in mm (from bottom edge of printable area)
            text: Text content
            font_size: Font size in points
            font_family: Font family name
            color: Text color (CSS color string)
            anchor: Text anchor: 'left', 'middle', 'right'
        """
        text_obj = Text(x, y, text, font_size, font_family, color, anchor)
        self.texts.append(text_obj)

    def _parse_color(self, color_str: str) -> Color:
        """Parse color string to ReportLab Color object."""
        color_map = {
            "black": black,
            "blue": blue,
            "green": green,
            "red": red,
            "darkred": Color(0.5, 0.0, 0.0),
            "darkblue": Color(0.0, 0.0, 0.5),
            "darkgreen": Color(0.0, 0.5, 0.0),
            "orange": Color(1.0, 0.5, 0.0),
            "yellow": Color(1.0, 1.0, 0.0),
            "lightgray": lightgrey,
            "lightgrey": lightgrey,
            "lightblue": Color(0.7, 0.8, 1.0),
            "lightgreen": Color(0.7, 1.0, 0.7),
        }

        if color_str in color_map:
            return color_map[color_str]
        elif color_str.startswith("#"):
            # Parse hex color
            hex_color = color_str[1:]
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                return Color(r, g, b)

        return black  # Default to black if unknown

    def _get_pdf_font(self, font_family: str) -> str:
        """Map font family to ReportLab font name."""
        font_map = {
            "arial": "Helvetica",
            "helvetica": "Helvetica",
            "times": "Times-Roman",
            "courier": "Courier",
            "default": "Helvetica",
        }
        return font_map.get(font_family.lower(), "Helvetica")

    def _calculate_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Calculate the bounding box of all objects.

        Returns:
            Tuple of (min_x, min_y, max_x, max_y) in mm
        """
        if not self.rectangles and not self.texts:
            return (0.0, 0.0, 0.0, 0.0)

        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        # Check rectangles
        for rect in self.rectangles:
            min_x = min(min_x, rect.x)
            min_y = min(min_y, rect.y)
            max_x = max(max_x, rect.x + rect.width)
            max_y = max(max_y, rect.y + rect.height)

        # Check text elements (approximate bounding box)
        for text in self.texts:
            # Approximate text dimensions based on font size
            # This is a rough estimate - actual text rendering may vary
            char_width = text.font_size * 0.6  # Approximate character width
            text_width = len(text.text) * char_width
            text_height = text.font_size * 1.2  # Approximate line height

            # Adjust position based on anchor
            if text.anchor == "left":
                text_min_x = text.x
                text_max_x = text.x + text_width
            elif text.anchor == "middle":
                text_min_x = text.x - text_width / 2
                text_max_x = text.x + text_width / 2
            else:  # right
                text_min_x = text.x - text_width
                text_max_x = text.x

            min_x = min(min_x, text_min_x)
            min_y = min(min_y, text.y)
            max_x = max(max_x, text_max_x)
            max_y = max(max_y, text.y + text_height)

        return (min_x, min_y, max_x, max_y)

    def _get_centering_offset(self) -> Tuple[float, float]:
        """
        Calculate the offset needed to center all objects on the page.

        Returns:
            Tuple of (offset_x, offset_y) in mm
        """
        min_x, min_y, max_x, max_y = self._calculate_bounding_box()

        # Calculate object dimensions
        object_width = max_x - min_x
        object_height = max_y - min_y

        # Calculate center position
        center_x = (self.printable_width - object_width) / 2
        center_y = (self.printable_height - object_height) / 2

        # Calculate offset needed to move objects to center
        offset_x = center_x - min_x
        offset_y = center_y - min_y

        return (offset_x, offset_y)

    def clear(self) -> None:
        """Clear all rectangles and text."""
        self.rectangles.clear()
        self.texts.clear()

    def to_svg(self, output_path: str, center_objects: bool = False) -> None:
        """
        Generate SVG file with accurate millimeter positioning.

        Args:
            output_path: Path to save the SVG file
            center_objects: If True, center all objects on the page
        """
        # Calculate centering offset if needed
        offset_x, offset_y = (0.0, 0.0)
        if center_objects:
            offset_x, offset_y = self._get_centering_offset()

        # Create SVG with A4 dimensions
        # SVG uses user units, we set viewBox to match A4 in mm
        dwg = svgwrite.Drawing(
            output_path,
            size=(f"{self.A4_WIDTH_MM}mm", f"{self.A4_HEIGHT_MM}mm"),
            viewBox=f"0 0 {self.A4_WIDTH_MM} {self.A4_HEIGHT_MM}",
        )

        # Add a border to show the printable area
        dwg.add(
            dwg.rect(
                insert=(self.margin_mm, self.margin_mm),
                size=(self.printable_width, self.printable_height),
                fill="none",
                stroke="lightgray",
                stroke_width=0.1,
            )
        )

        # Add rectangles
        for rect in self.rectangles:
            # Apply centering offset
            adjusted_x = rect.x + offset_x
            adjusted_y = rect.y + offset_y

            # Convert from bottom-left coordinate system to SVG's top-left system
            # SVG Y coordinate = total_height - (user_y + rect_height)
            svg_x = adjusted_x + self.margin_mm
            svg_y = self.A4_HEIGHT_MM - (adjusted_y + self.margin_mm + rect.height)

            dwg.add(
                dwg.rect(
                    insert=(svg_x, svg_y),
                    size=(rect.width, rect.height),
                    fill=rect.fill_color,
                    stroke=rect.stroke_color,
                    stroke_width=rect.stroke_width,
                )
            )

        # Add text elements
        for text in self.texts:
            # Apply centering offset
            adjusted_x = text.x + offset_x
            adjusted_y = text.y + offset_y

            # Convert from bottom-left coordinate system to SVG's top-left system
            # For text, we use the baseline position
            svg_x = adjusted_x + self.margin_mm
            svg_y = self.A4_HEIGHT_MM - (adjusted_y + self.margin_mm)

            # Map anchor to SVG text-anchor
            text_anchor = "start" if text.anchor == "left" else text.anchor
            if text.anchor == "right":
                text_anchor = "end"

            dwg.add(
                dwg.text(
                    text.text,
                    insert=(svg_x, svg_y),
                    font_size=f"{text.font_size}pt",
                    font_family=text.font_family,
                    fill=text.color,
                    text_anchor=text_anchor,
                )
            )

        dwg.save()

    def to_pdf(self, output_path: str, center_objects: bool = False) -> None:
        """
        Generate PDF file with accurate millimeter positioning.

        Args:
            output_path: Path to save the PDF file
            center_objects: If True, center all objects on the page
        """
        # Calculate centering offset if needed
        offset_x, offset_y = (0.0, 0.0)
        if center_objects:
            offset_x, offset_y = self._get_centering_offset()

        # Create PDF with A4 size
        c = canvas.Canvas(output_path, pagesize=A4)

        # ReportLab uses points (72 DPI) by default
        # We use the mm unit multiplier for accurate positioning

        # Draw border of printable area
        c.setStrokeColor(lightgrey)
        c.setLineWidth(0.1 * mm)
        c.rect(
            self.margin_mm * mm,
            (self.A4_HEIGHT_MM - self.margin_mm - self.printable_height) * mm,
            self.printable_width * mm,
            self.printable_height * mm,
            fill=0,
        )

        # Add rectangles
        for rect in self.rectangles:
            # Apply centering offset
            adjusted_x = rect.x + offset_x
            adjusted_y = rect.y + offset_y

            # PDF coordinates (ReportLab uses bottom-left origin, same as our system)
            pdf_x = (adjusted_x + self.margin_mm) * mm
            pdf_y = (adjusted_y + self.margin_mm) * mm

            # Set colors
            if rect.fill_color != "none":
                c.setFillColor(self._parse_color(rect.fill_color))
            c.setStrokeColor(self._parse_color(rect.stroke_color))
            c.setLineWidth(rect.stroke_width * mm)

            # Draw rectangle
            fill_mode = 1 if rect.fill_color != "none" else 0
            c.rect(pdf_x, pdf_y, rect.width * mm, rect.height * mm, fill=fill_mode)

        # Add text elements
        for text in self.texts:
            # Apply centering offset
            adjusted_x = text.x + offset_x
            adjusted_y = text.y + offset_y

            # PDF coordinates (ReportLab uses bottom-left origin, same as our system)
            pdf_x = (adjusted_x + self.margin_mm) * mm
            pdf_y = (adjusted_y + self.margin_mm) * mm

            # Set text properties
            pdf_font = self._get_pdf_font(text.font_family)
            c.setFont(pdf_font, text.font_size)
            c.setFillColor(self._parse_color(text.color))

            # Draw text with appropriate alignment
            if text.anchor == "left":
                c.drawString(pdf_x, pdf_y, text.text)
            elif text.anchor == "middle":
                c.drawCentredString(pdf_x, pdf_y, text.text)
            elif text.anchor == "right":
                c.drawRightString(pdf_x, pdf_y, text.text)

        c.save()

    def svg_to_pdf_weasyprint(self, svg_path: str, pdf_path: str) -> None:
        """
        Convert SVG to PDF using WeasyPrint (alternative method).

        Args:
            svg_path: Path to the SVG file
            pdf_path: Path to save the PDF file
        """
        try:
            import weasyprint

            # Create HTML wrapper for the SVG
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        margin: 0;
                        padding: 0;
                    }}
                    svg {{
                        width: 210mm;
                        height: 297mm;
                    }}
                </style>
            </head>
            <body>
                <object data="{svg_path}" type="image/svg+xml" width="210mm" height="297mm">
                    <img src="{svg_path}" alt="SVG Image" style="width: 210mm; height: 297mm;" />
                </object>
            </body>
            </html>
            """

            # Convert to PDF
            html_doc = weasyprint.HTML(
                string=html_content, base_url=os.path.dirname(svg_path)
            )
            html_doc.write_pdf(pdf_path)

        except ImportError:
            raise ImportError("WeasyPrint not available. Use to_pdf() method instead.")

    def print_info(self) -> None:
        """Print information about the current setup."""
        print(f"Calibrated Paper Tool")
        print(f"====================")
        print(f"Paper size: {self.A4_WIDTH_MM}mm × {self.A4_HEIGHT_MM}mm (A4)")
        print(f"Margin: {self.margin_mm}mm")
        print(f"Printable area: {self.printable_width}mm × {self.printable_height}mm")
        print(f"Rectangles: {len(self.rectangles)}")

        for i, rect in enumerate(self.rectangles):
            print(f"  [{i}] {rect.width}×{rect.height}mm at ({rect.x}, {rect.y})")

        print(f"Text elements: {len(self.texts)}")
        for i, text in enumerate(self.texts):
            print(
                f"  [{i}] '{text.text}' at ({text.x}, {text.y}) - {text.font_size}pt {text.font_family}"
            )

    def save_both(self, base_path: str, center_objects: bool = False) -> None:
        """
        Save both SVG and PDF files with the same base name.

        Args:
            base_path: Base path for output files (without extension)
            center_objects: If True, center all objects on the page
        """
        svg_path = base_path + ".svg"
        pdf_path = base_path + ".pdf"
        self.to_svg(svg_path, center_objects)
        self.to_pdf(pdf_path, center_objects)

        if center_objects:
            min_x, min_y, max_x, max_y = self._calculate_bounding_box()
            offset_x, offset_y = self._get_centering_offset()
            print(f"Objects centered: offset ({offset_x:.1f}, {offset_y:.1f})mm")

        print(f"Files saved: {svg_path} and {pdf_path}")


def create_test_document() -> None:
    """
    Create a test document with two rectangles as specified.
    Uses CAD/3D printing coordinate system: origin at bottom-left, Y+ goes up.
    """
    # Create calibrated paper instance
    paper = CalibratedPaper(margin_mm=15.0)  # 15mm margin

    # Add two rectangles as requested
    # Note: With bottom-left origin, Y+ goes up
    paper.add_rectangle(
        x=20.0,
        y=20.0,  # Position: 20mm from left, 20mm from bottom
        width=10.0,
        height=30.0,  # 10×30mm rectangle
        fill_color="lightblue",
        stroke_color="blue",
        stroke_width=0.2,
    )

    paper.add_rectangle(
        x=50.0,
        y=40.0,  # Position: 50mm from left, 40mm from bottom
        width=7.0,
        height=40.0,  # 7×40mm rectangle
        fill_color="lightgreen",
        stroke_color="green",
        stroke_width=0.2,
    )

    # Add some text labels
    paper.add_text(
        x=20.0,
        y=55.0,  # Position above first rectangle
        text="10×30mm",
        font_size=8,
        color="blue",
        anchor="left",
    )

    paper.add_text(
        x=50.0,
        y=85.0,  # Position above second rectangle
        text="7×40mm",
        font_size=8,
        color="green",
        anchor="left",
    )

    paper.add_text(
        x=100.0,
        y=20.0,  # Centered text
        text="Test Document",
        font_size=16,
        color="black",
        anchor="middle",
    )

    # Print info
    paper.print_info()

    # Generate output files
    output_dir = "/tmp"
    svg_path = os.path.join(output_dir, "test_rectangles.svg")
    pdf_path = os.path.join(output_dir, "test_rectangles.pdf")

    print(f"\nGenerating files...")
    paper.to_svg(svg_path)
    paper.to_pdf(pdf_path)

    print(f"SVG saved to: {svg_path}")
    print(f"PDF saved to: {pdf_path}")

    # Also demonstrate centering
    centered_svg = os.path.join(output_dir, "test_rectangles_centered.svg")
    centered_pdf = os.path.join(output_dir, "test_rectangles_centered.pdf")

    paper.to_svg(centered_svg, center_objects=True)
    paper.to_pdf(centered_pdf, center_objects=True)

    print(f"Centered SVG saved to: {centered_svg}")
    print(f"Centered PDF saved to: {centered_pdf}")

    print(f"\nCoordinate System:")
    print(f"- Origin (0,0) at bottom-left of printable area")
    print(f"- X+ goes to the right")
    print(f"- Y+ goes up")
    print(f"- Compatible with CAD and 3D printing conventions")

    print(f"\nFor accurate printing:")
    print(f"1. Open the PDF in your PDF viewer")
    print(f"2. Print with 'Actual Size' or '100% scale' (no scaling)")
    print(f"3. Measure the rectangles with a ruler to verify accuracy")


if __name__ == "__main__":
    create_test_document()
