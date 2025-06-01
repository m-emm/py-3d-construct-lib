import math

import numpy as np


def normalize_edge(a, b):
    return tuple(sorted((a, b)))


def triangle_edges(tri):
    return [(tri[i], tri[(i + 1) % 3]) for i in range(3)]


def compute_triangle_normal(v0, v1, v2):
    return np.cross(v1 - v0, v2 - v0)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def fibonacci_sphere(samples=100):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append(np.array([x, y, z]))
    return points
