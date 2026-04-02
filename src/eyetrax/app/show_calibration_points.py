"""
Visualise calibration point layouts for report purposes.
Press any key (or wait) to cycle through the three layouts. ESC to quit.

Layouts shown:
  1. Original 5-point
  2. Single vertical line (14 points: 5 corners + 9 centre)
  3. Dual vertical lines  (23 points: 5 corners + 9 left + 9 right)
"""

import cv2
import numpy as np


# ── point computation (mirrors the actual calibration logic) ──────────────────

def compute_grid_points(order, sw, sh, margin_ratio=0.10):
    if not order:
        return []
    max_r = max(r for r, _ in order)
    max_c = max(c for _, c in order)
    mx, my = int(sw * margin_ratio), int(sh * margin_ratio)
    gw, gh = sw - 2 * mx, sh - 2 * my
    step_x = 0 if max_c == 0 else gw / max_c
    step_y = 0 if max_r == 0 else gh / max_r
    return [(mx + int(c * step_x), my + int(r * step_y)) for r, c in order]


NINE_POINT = [(1,1),(0,0),(2,0),(0,2),(2,2),(1,0),(0,1),(2,1),(1,2)]


def get_layouts(sw, sh):
    nine_pts = compute_grid_points(NINE_POINT, sw, sh, margin_ratio=0.10)
    corners  = compute_grid_points([(1,1),(0,0),(2,0),(0,2),(2,2)], sw, sh, margin_ratio=0.10)

    num_v = 9
    ys = [int(sh * (0.05 + i * (0.90 / (num_v - 1)))) for i in range(num_v)]

    center_line  = [(sw // 2, y) for y in ys]
    left_line    = [(sw // 2 - 75, y) for y in ys]
    right_line   = [(sw // 2 + 75, y) for y in ys]

    return [
        {
            "title": "9-Point",
            "groups": [{"points": nine_pts, "color": (0, 255, 0)}],
        },
        {
            "title": "Single Vertical Line",
            "groups": [
                {"points": corners,     "color": (0, 255, 0)},
                {"points": center_line, "color": (0, 255, 0)},
            ],
        },
        {
            "title": "Dual Vertical Lines",
            "groups": [
                {"points": corners,    "color": (0, 255, 0)},
                {"points": left_line,  "color": (0, 255, 0)},
                {"points": right_line, "color": (0, 255, 0)},
            ],
        },
    ]


# ── drawing ───────────────────────────────────────────────────────────────────

DOT_R   = 10   # filled dot radius
CROSS_R = 16   # crosshair arm length


def draw_layout(sw, sh, layout):
    canvas = np.zeros((sh, sw, 3), dtype=np.uint8)

    # faint grid lines for reference
    for frac in [0.10, 0.50, 0.90]:
        xg, yg = int(sw * frac), int(sh * frac)
        cv2.line(canvas, (xg, 0),  (xg, sh),  (30, 30, 30), 1)
        cv2.line(canvas, (0,  yg), (sw,  yg), (30, 30, 30), 1)

    # draw each group of points
    for g in layout["groups"]:
        col = (0, 255, 0)
        for (px, py) in g["points"]:
            # crosshair
            cv2.line(canvas, (px - CROSS_R, py), (px + CROSS_R, py), col, 1)
            cv2.line(canvas, (px, py - CROSS_R), (px, py + CROSS_R), col, 1)
            # filled dot
            cv2.circle(canvas, (px, py), DOT_R, col, -1)

    return canvas


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    win = "Calibration Point Layouts"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # get actual screen size from a throwaway frame
    tmp = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imshow(win, tmp)
    cv2.waitKey(1)
    # use a generous default; replace with your screen res if needed
    sw, sh = 1920, 1080

    # try to detect screen size via the window
    try:
        from eyetrax.utils.screen import get_screen_size
        sw, sh = get_screen_size()
    except Exception:
        pass

    layouts = get_layouts(sw, sh)

    for i, layout in enumerate(layouts):
        canvas = draw_layout(sw, sh, layout)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
