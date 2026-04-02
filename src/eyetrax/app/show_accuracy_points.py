"""
Visualise the accuracy test point layout for report purposes.
Shows all 7 points at once. Press any key or ESC to quit.
"""

import cv2
import numpy as np


def main():
    win = "Accuracy Test Layout"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        from eyetrax.utils.screen import get_screen_size
        sw, sh = get_screen_size()
    except Exception:
        sw, sh = 1920, 1080

    # Mirror the exact point computation from vertical_accuracy.py
    margin_ratio = 0.10
    my = int(sh * margin_ratio)
    gh = sh - 2 * my
    num_points = 7
    step_y = gh / (num_points - 1)
    points = [(sw // 2, my + int(i * step_y)) for i in range(num_points)]

    canvas = np.zeros((sh, sw, 3), dtype=np.uint8)

    # faint vertical centre line for reference
    cv2.line(canvas, (sw // 2, 0), (sw // 2, sh), (30, 30, 30), 1)

    for px, py in points:
        cv2.circle(canvas, (px, py), 10, (0, 255, 0), -1)

    cv2.imshow(win, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
