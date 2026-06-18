import argparse
import sys
import time

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time object detection using YOLOv5 and OpenCV"
    )
    parser.add_argument(
        "--weights", type=str, default="yolov5s.pt",
        help="Path to model weights (.pt file)"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Webcam index (e.g. 0) or path to a video/image file"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for detections (0-1)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save the annotated output video (optional)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without opening a display window (useful for headless runs)"
    )
    return parser.parse_args()


def resolve_source(source_str):
    """Return an int for webcam indices, otherwise treat as a file path."""
    return int(source_str) if source_str.isdigit() else source_str


def main():
    args = parse_args()

    print(f"Loading model weights from '{args.weights}'...")
    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"Error: failed to load weights '{args.weights}' ({e})")
        sys.exit(1)

    source = resolve_source(args.source)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: could not open video source '{args.source}'")
        sys.exit(1)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps_in, (width, height))

    prev_time = time.time()
    print("Detection running. Press ESC to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or failed to grab frame.")
                break

            results = model.predict(frame, conf=args.conf, verbose=False)
            annotated = results[0].plot()

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            if writer:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Object Detection", annotated)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
