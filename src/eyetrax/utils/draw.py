from __future__ import annotations

import cv2
import numpy as np


def draw_cursor(
    canvas,
    x: int,
    y: int,
    alpha: float,
    *,
    radius_outer: int = 30,
    radius_inner: int = 25,
    color_outer: tuple[int, int, int] = (0, 0, 255),
    color_inner: tuple[int, int, int] = (255, 255, 255),
):
    if alpha <= 0.0:
        return canvas

    overlay = canvas.copy()
    cv2.circle(overlay, (int(x), int(y)), radius_outer, color_outer, -1)
    if radius_inner > 0:
        cv2.circle(overlay, (int(x), int(y)), radius_inner, color_inner, -1)

    cv2.addWeighted(overlay, alpha * 0.6, canvas, 1 - alpha * 0.6, 0, canvas)
    return canvas


def make_thumbnail(
    frame,
    *,
    size: tuple[int, int] = (320, 240),
    border: int = 2,
    border_color: tuple[int, int, int] = (255, 255, 255),
):
    img = cv2.resize(frame, size)
    return cv2.copyMakeBorder(
        img,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )


def draw_scan_path(
    canvas,
    path_points: list[tuple[int, int]],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    fade_alpha: bool = True,
    max_points: int | None = None,
):
    """
    Draw the eye scan path on the canvas.
    
    Args:
        canvas: The canvas to draw on
        path_points: List of (x, y) tuples representing the gaze path
        color: RGB color for the path (default: green)
        thickness: Line thickness
        fade_alpha: If True, older points fade out (becomes more transparent)
        max_points: Maximum number of points to draw (None = draw all)
    """
    if not path_points or len(path_points) < 2:
        return canvas
    
    # Limit the number of points to draw if specified
    points_to_draw = path_points
    if max_points is not None and len(path_points) > max_points:
        points_to_draw = path_points[-max_points:]
    
    # Draw lines connecting consecutive points
    for i in range(len(points_to_draw) - 1):
        pt1 = points_to_draw[i]
        pt2 = points_to_draw[i + 1]
        
        # Calculate alpha for fading effect
        if fade_alpha and max_points is not None:
            # Older points are more transparent
            alpha = (i + 1) / max_points
            alpha = max(0.3, alpha)  # Minimum alpha of 0.3
            # Blend color with background
            blended_color = tuple(int(c * alpha) for c in color)
        else:
            blended_color = color
        
        cv2.line(canvas, pt1, pt2, blended_color, thickness)
    
    # Draw a small circle at the most recent point
    if points_to_draw:
        latest_point = points_to_draw[-1]
        cv2.circle(canvas, latest_point, 3, color, -1)
    
    return canvas
