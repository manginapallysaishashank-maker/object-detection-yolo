# YOLOv5 Real-Time Object Detection

Real-time object detection using YOLOv5 and OpenCV, with a live FPS overlay, adjustable confidence threshold, and support for webcam, video files, or saved output.

## Features
- Detection from a webcam or a video/image file
- Live FPS counter overlaid on the output
- Adjustable confidence threshold via command line
- Optional saving of annotated output to a video file
- Headless mode for running without a display window (e.g. on a server)
- Clear error handling for missing weights or unavailable video sources

## Demo
*(Add a screenshot or short GIF of the detection output here.)*

## Tech Stack
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv5

## Project Structure
```
object-detection-yolo/
├── detect.py          # Main detection script
├── yolov5s.pt          # Pretrained YOLOv5 small model weights
├── requirements.txt    # Python dependencies
├── README.md
└── .gitignore
```

## Installation
```bash
git clone https://github.com/manginapallysaishashank-maker/object-detection-yolo.git
cd object-detection-yolo
pip install -r requirements.txt
```

## Usage

Run with the default webcam:
```bash
python detect.py
```

Run with a custom confidence threshold:
```bash
python detect.py --conf 0.4
```

Run on a video file and save the annotated result:
```bash
python detect.py --source path/to/video.mp4 --output result.mp4
```

Run headless, without opening a display window:
```bash
python detect.py --no-display --output result.mp4
```

## Command-Line Arguments
| Argument       | Description                                       | Default     |
|----------------|----------------------------------------------------|-------------|
| `--weights`    | Path to model weights file                         | `yolov5s.pt`|
| `--source`     | Webcam index or path to a video/image file         | `0`         |
| `--conf`       | Confidence threshold (0-1)                          | `0.25`      |
| `--output`     | Path to save the annotated output video            | `None`      |
| `--no-display` | Run without opening a display window               | `False`     |

## How It Works
The script loads the YOLOv5 model directly from the local `yolov5s.pt` weights using the Ultralytics package. Each frame from the chosen source is passed through the model, annotated with bounding boxes, class labels, and a live FPS counter, then either displayed in a window, written to an output video file, or both.

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
