# MediaPipe Mouse Control

This is a MediaPipe pipeline for controlling the computer mouse using hand gestures. The pipeline uses a webcam or video file as input, detects the hand and fingers in the image, and maps the finger movements to cursor movements on the screen.

## Requirements

- MediaPipe
- A webcam or video file for input
- A computer running a supported operating system (Windows, Mac, Linux)

## Usage

1. Install MediaPipe following the instructions on the [MediaPipe website](https://mediapipe.readthedocs.io/en/latest/install.html).

2. Clone this repository and navigate to the `mouse_control` directory.

3. Run the pipeline using the following command:
```bash
mediapipe run --calculator_graph_config_file=mouse_control.pbtxt --input_stream=input_video:<path_to_input_video> --output_stream=output_video:<path_to_output_video>
