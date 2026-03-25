# YOLO Vehicle Detection 🚗💨

A high-performance vehicle detection system using **YOLOv10x** for images and **YOLOv8n** for videos. This project is capable of detecting cars, trucks, buses, and motorbikes with high accuracy and speed.

## ✨ Features
- **Precise Image Detection**: Uses YOLOv10x for superior detection quality in static images.
- **Fast Video Detection**: Uses YOLOv8n to maintain a high frame rate during video processing.
- **Easy-to-use CLI**: A standalone script for quick detection runs.
- **Jupyter Notebook Support**: Robust notebook for interactive analysis.
- **Automated Output Management**: Results are automatically saved to an `outputs/` directory.

## 🛠 Prerequisites
- **Python 3.8+**
- **Hardware**: GPU (recommended for faster processing) or CPU.

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/omerfarooq223/YOLO_Vehicle_Detection.git
    cd YOLO_Vehicle_Detection
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📖 Usage

### Using the Python Script (`detect.py`)
Run detection from the terminal:

- **Image Detection**:
  ```bash
  python detect.py --source sample_data/sample_1.jpeg --type image
  ```

- **Video Detection**:
  ```bash
  python detect.py --source path/to/your/video.mp4 --type video
  ```

*Results will be saved in the `outputs/` directory.*

### Using the Jupyter Notebook (`detect_vehicles.ipynb`)
For an interactive experience, open the notebook:
```bash
jupyter notebook detect_vehicles.ipynb
```
Follow the steps in the notebook to run detections on your data.

## 📂 Project Structure
- `detect.py`: Standalone CLI tool for vehicle detection.
- `detect_vehicles.ipynb`: Interactive Jupyter notebook.
- `requirements.txt`: Python package dependencies.
- `sample_data/`: Contains sample images for testing.
- `outputs/`: (Created upon running) Stores processed results.

## 📊 Sample Detections
You can find test images in the `sample_data/` folder:
- `sample_1.jpeg`
- `sample_2.jpeg`
- `sample_3.jpeg`
- `sample_4.jpeg`

---
*Developed by Muhammad Umar Farooq.*
