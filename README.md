# Cow Detection using YOLOv5 🌱🐄

This repository contains code for detecting cows in videos using the YOLOv5 object detection model. It utilizes the `ultralytics/yolov5` PyTorch hub model for inference.

## Installation

1. Clone the repository: 🛠️

```bash
git clone https://github.com/HosseinTahami/CowDetection.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage 🐄💻

1. Place your input video file in the repository directory.

2. Open the detect_cows.py file and modify the following variables:

      ⚫️ video_path: Path to your input video file.

      ⚫️ output_path: Path to save the output video file.

3. Run the script:

```bash
python detect_cows.py
```

The script will process the video, detect cows using YOLOv5, draw bounding boxes around them, and save the output video.

4. The output video will be saved in the specified output_path.


## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License 📝

This project is licensed under the MIT License.
