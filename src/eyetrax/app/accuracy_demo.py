"""
Standalone application to run vertical accuracy test
"""
import os

from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_lissajous_calibration,
    run_vertical_enhanced_calibration,  # NEW
)
from eyetrax.calibration.vertical_accuracy import run_vertical_accuracy_test
from eyetrax.cli import parse_common_args
from eyetrax.gaze import GazeEstimator


def run_accuracy_test():
    """
    Run the vertical accuracy test after calibration
    """
    args = parse_common_args()
    
    camera_index = args.camera
    calibration_method = args.calibration
    
    gaze_estimator = GazeEstimator(model_name=args.model)
    
    # Load existing model or run calibration
    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[accuracy_test] Loaded gaze model from {args.model_file}")
    else:
        print("[accuracy_test] Running calibration first...")
        
        if calibration_method == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "vertical":  # NEW
            run_vertical_enhanced_calibration(gaze_estimator, camera_index=camera_index)
        else:
            run_lissajous_calibration(gaze_estimator, camera_index=camera_index)
        
        print("[accuracy_test] Calibration complete!")
    
    # Run accuracy test
    print("[accuracy_test] Starting vertical accuracy test...")
    results = run_vertical_accuracy_test(gaze_estimator, camera_index=camera_index)
    
    if results:
        print("[accuracy_test] Test completed successfully!")
        
        # Optionally save results
        if args.model_file:
            results_file = args.model_file.replace('.pkl', '_accuracy.txt')
            with open(results_file, 'w') as f:
                f.write("VERTICAL ACCURACY TEST RESULTS\n")
                f.write("="*50 + "\n")
                f.write(f"Mean Vertical Error: {results['mean_vertical_error']:.2f} pixels\n")
                f.write(f"Vertical RMSE: {results['rmse_vertical']:.2f} pixels\n")
                f.write(f"Std Dev (Vertical): {results['std_vertical_error']:.2f} pixels\n")
                f.write(f"Mean Horizontal Error: {results['mean_horizontal_error']:.2f} pixels\n")
                f.write("="*50 + "\n")
            print(f"[accuracy_test] Results saved to {results_file}")
    else:
        print("[accuracy_test] Test was cancelled or failed")


if __name__ == "__main__":
    run_accuracy_test()
