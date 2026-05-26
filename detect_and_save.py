import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run YOLOv8 detection on an image or video and save annotated output to a new file.'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to the input image or video file.'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Path to save the annotated output file. If omitted, a new file name is generated.'
    )
    parser.add_argument(
        '--model', '-m',
        default='weights/best.pt',
        help='Path to the YOLO model file. Default: weights/best.pt'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.35,
        help='Confidence threshold for detections. Default: 0.35'
    )
    return parser.parse_args()


def is_video_file(path: Path) -> bool:
    video_ext = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.webm'}
    return path.suffix.lower() in video_ext


def make_output_path(input_path: Path, output_path: str | None) -> Path:
    if output_path:
        return Path(output_path)

    suffix = input_path.suffix
    stem = input_path.stem
    parent = input_path.parent
    return parent / f'{stem}_annotated{suffix}'


def annotate_image(model: YOLO, image_path: Path, output_path: Path, conf: float):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    results = model.predict(source=image, conf=conf, save=False)
    if len(results) == 0:
        raise RuntimeError('Model returned no results for the provided image.')

    annotated = results[0].plot()
    success = cv2.imwrite(str(output_path), annotated)
    if not success:
        raise IOError(f'Failed to write annotated image to: {output_path}')

    print(f'Saved annotated image to: {output_path}')


def annotate_video(model: YOLO, video_path: Path, output_path: Path, conf: float):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f'Could not open video: {video_path}')

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        results = model.predict(source=frame, conf=conf, save=False)
        annotated = results[0].plot() if len(results) else frame
        writer.write(annotated)
        frame_count += 1

    capture.release()
    writer.release()
    print(f'Saved annotated video with {frame_count} frames to: {output_path}')


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input file does not exist: {input_path}')

    output_path = make_output_path(input_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading model: {args.model}')
    model = YOLO(str(args.model))

    if is_video_file(input_path):
        annotate_video(model, input_path, output_path, args.conf)
    else:
        annotate_image(model, input_path, output_path, args.conf)


if __name__ == '__main__':
    main()
