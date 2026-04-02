import argparse


def parse_common_args():

    parser = argparse.ArgumentParser(description="Common Gaze Estimation Arguments")

    parser.add_argument(
        "--filter",
        choices=["kalman", "kde", "none"],
        default="none",
        help="Select the filter to apply to gaze estimation, options are 'kalman', 'kde', or 'none'",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for video capture, default is 0 (first camera)",
    )
    parser.add_argument(
        "--calibration",
        choices=["9p", "5p", "lissajous", "vertical"],
        default="9p",
        help="Calibration method for gaze estimation, options are '9p', '5p', or 'lissajous'",
    )
    parser.add_argument(
        "--background",
        type=str,
        default=None,
        help="Path to a custom background image (optional)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence level for KDE smoothing, range 0 to 1",
    )
    parser.add_argument(
        "--model",
        default="ridge",
        help="The machine learning model to use for gaze estimation, default is 'ridge'",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help="Path to a previously-trained gaze model",
    )
    parser.add_argument(
        "--scan-path",
        action="store_true",
        default=False,
        help="Enable visualization of eye scan path on screen",
    )
    parser.add_argument(
        "--scan-path-max",
        type=int,
        default=500,
        help="Maximum number of points to keep in scan path (default: 500)",
    )
    parser.add_argument(
        "--scan-path-log",
        type=str,
        default=None,
        help="Path to save scan path log file (CSV format). If not specified, saves to 'scan_path_<timestamp>.csv' on exit",
    )
    parser.add_argument(
        "--cursor",
        action="store_true",
        default=False,
        help="Enable gaze cursor dot and scan path trail overlay",
    )
    parser.add_argument(
        "--multi-pose",
        action="store_true",
        default=False,
        help="Enable multi-pose calibration (collect samples while moving head for better robustness)",
    )
    parser.add_argument(
        "--landmark-alpha",
        type=float,
        default=0.7,
        help="EMA alpha for raw landmark smoothing (1.0 = no smoothing, lower = more smoothing, default: 0.7)",
    )
    parser.add_argument(
        "--feature-alpha",
        type=float,
        default=None,
        help="EMA alpha for feature vector smoothing (None = disabled, 0.5 = moderate smoothing)",
    )
    parser.add_argument(
        "--pose-damping",
        type=float,
        default=0.0,
        help="Head pose drift compensation (0.0 = off, 1.0 = fully clamp to calibration pose, default: 0.0)",
    )
    parser.add_argument(
        "--multi-position",
        action="store_true",
        default=False,
        help="Run calibration at 3 vertical face positions (above/center/below camera) for position-robust gaze estimation",
    )
    parser.add_argument(
        "--single-column",
        action="store_true",
        default=False,
        help="Use a single center vertical line instead of two lines (vertical calibration only)",
    )
    parser.add_argument(
        "--save-calibration",
        type=str,
        default=None,
        metavar="PATH",
        help="After calibrating, save the model to this file (e.g. my_calib.pkl)",
    )

    return parser.parse_args()
