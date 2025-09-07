# py-3d-construct-lib

A Python library for 3D geometric construction, mesh manipulation, and spherical tools.

## Overview

`py-3d-construct-lib` provides a comprehensive set of tools for working with 3D geometry, particularly focused on:

- **Spherical geometry and coordinate transformations**
- **3D mesh partitioning and manipulation**
- **Geometric construction utilities**
- **Face point cloud generation and processing**
- **Calibrated paper tools for precise printing**

## Features

### Coordinate System Transformations

The library provides powerful tools for rigid body transformations between coordinate systems:

- **`coordinate_system_transform`**: Compute rigid transformations (rotation + translation) to align one coordinate system to another
- **`coordinate_system_transformation_function`**: Create transformation functions that can be applied to objects using custom rotation and translation generators
- **`coordinate_system_transform_to_matrix`**: Convert transformation parameters to 4x4 homogeneous transformation matrices
- **`matrix_to_coordinate_system_transform`**: Extract transformation parameters from 4x4 matrices

These functions work with coordinate systems defined by origin, up vector, and out vector, using Gram-Schmidt orthogonalization for robust transformations.

### Spherical Tools
- Convert between Cartesian and spherical coordinates
- Spherical triangle manipulation and shrinking
- Rotation matrix calculations from vectors
- Spherical cap filtering and geometric operations

### Geometric Construction
- Icosahedron geometry generation
- Triangle mesh utilities and validation
- Rigid body transformation validation
- Fibonacci sphere point distribution
- Edge normalization and triangle operations

### Mesh Processing
- Mesh partitioning algorithms
- Face point cloud generation
- Region edge feature detection
- Connector hint computation
- Shell mapping and collinear connector merging

### Calibrated Paper Output
- Millimeter-precise geometry rendering
- SVG and PDF export with accurate positioning
- A4 paper layout with customizable margins
- CAD-compatible coordinate system (bottom-left origin)

## Installation

```bash
pip install py-3d-construct-lib
```

## Quick Start

### Coordinate System Transformations

```python
import numpy as np
from py_3d_construct_lib.spherical_tools import (
    coordinate_system_transform,
    coordinate_system_transform_to_matrix,
    matrix_to_coordinate_system_transform
)

# Define two coordinate systems
origin_a = np.array([0, 0, 0])
up_a = np.array([0, 0, 1])      # Z-up
out_a = np.array([1, 0, 0])     # X-out

origin_b = np.array([10, 5, 2])
up_b = np.array([1, 0, 0])      # X-up  
out_b = np.array([0, 1, 0])     # Y-out

# Compute transformation from system A to system B
transform = coordinate_system_transform(origin_a, up_a, out_a, 
                                      origin_b, up_b, out_b)

# Convert to 4x4 transformation matrix
matrix = coordinate_system_transform_to_matrix(transform)

# Extract transformation back from matrix
recovered_transform = matrix_to_coordinate_system_transform(matrix)

print(f"Rotation angle: {transform['rotation_angle']:.3f} radians")
print(f"Rotation axis: {transform['rotation_axis']}")
print(f"Translation: {transform['translation']}")
```

### Basic Spherical Operations

```python
import numpy as np
from py_3d_construct_lib.spherical_tools import (
    spherical_to_cartesian_jackson,
    cartesian_to_spherical_jackson
)

# Convert spherical to cartesian coordinates
sph_coords = (1.0, np.pi/4, np.pi/6)  # (r, theta, phi)
cartesian = spherical_to_cartesian_jackson(sph_coords)

# Convert back to spherical
spherical = cartesian_to_spherical_jackson(cartesian)
```

### Creating Geometric Shapes

```python
from py_3d_construct_lib.geometries import create_icosahedron_geometry

# Create an icosahedron with radius 2.0
vertices, faces = create_icosahedron_geometry(radius=2.0)
print(f"Created icosahedron with {len(vertices)} vertices and {len(faces)} faces")
```

### Calibrated Paper for Precise Printing

```python
from py_3d_construct_lib.calibrated_paper import CalibratedPaper

# Create calibrated paper with 10mm margins
paper = CalibratedPaper(margin_mm=10.0)

# Add rectangles with precise positioning
paper.add_rectangle(x=20.0, y=20.0, width=10.0, height=30.0, 
                   fill_color="lightblue", stroke_color="blue")

# Export to SVG and PDF
paper.save_svg("output.svg")
paper.save_pdf("output.pdf")
paper.print_info()
```

### Fibonacci Sphere Point Distribution

```python
from py_3d_construct_lib.construct_utils import fibonacci_sphere

# Generate 100 evenly distributed points on a sphere
points = fibonacci_sphere(num_points=100, radius=1.0)
```

## Module Overview

| Module | Description |
|--------|-------------|
| `geometries` | Basic geometric shape generation (icosahedron, etc.) |
| `spherical_tools` | Spherical coordinate systems and transformations |
| `construct_utils` | General construction utilities and validation |
| `face_point_cloud` | Point cloud generation on mesh faces |
| `mesh_partition` | Advanced mesh partitioning algorithms |
| `calibrated_paper` | Precise geometric output for printing |
| `connector_utils` | Connector hint computation and merging |
| `transformed_region_view` | Region transformation and viewing |

## Requirements

- Python 3.8+
- NumPy
- SciPy
- NetworkX
- svgwrite
- reportlab

## Development

This project uses PyScaffold for project structure and management.

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests

# Format code with precommit script (runs isort + black on src/ and tests/, prettier on workflows)
./precommit.sh

# Or manually format code
black src/
```

## License

[Add your license information here]

## Contributing

[Add contributing guidelines here]