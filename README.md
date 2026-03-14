# CrowdAware Offline (Raspberry Pi 4B + Heltec LoRa V3)
An offline, edge‑run version of CrowdAware consisting of:
- A Raspberry Pi 4B visualizer and parser for thermal + RGB data
- An Arduino/ESP32 (Heltec LoRa V3) node that reads the MLX90640 thermal sensor, performs lightweight on‑device processing, and streams binary frames
- YOLO26n exported to ONNX for lightweight detection on Pi
This repository is designed for fully offline deployment for data collection and testing of algoeithm accuracy.

<img width="1684" height="738" alt="image" src="https://github.com/user-attachments/assets/e6ff826f-2262-4b1c-a056-4e64b7b43e2d" />

## Repository Structure
```
├── node/                 # Arduino/ESP32 code for Heltec LoRa V3
│   ├── config.h
│   ├── mlx_sensor.cpp/.h
│   ├── MLX90640_API.cpp/.h
│   ├── MLX90640_I2C_Driver.cpp/.h
│   ├── thermal_image_processor.cpp/.h
│   ├── node.ino
│   └── ...
└── python_parser/        # Raspberry Pi 4B visualizer + parser
│   ├── parser_pi.py
│   └── parser_win.py
└── evaluation/           # Compute the accuracy of the thermal camera by comparing with the RGB (YOLOv26n)
    └── accuracy_calculation.py
```


## System Overview
### Heltec LoRa V3 Node (ESP32)
- Reads 32×24 thermal frames from MLX90640
- Performs background subtraction + blob detection
- Outputs binary‑encoded frames over serial or LoRa
- Designed for 1 MHz I²C operation (configurable)
- Uses Melexis’ official MLX90640 API (included)
### Raspberry Pi 4B Visualizer
- Receives binary frames from the node
- Converts them into temperature maps
- Runs YOLO26n (ONNX) on RGB camera frames (RPi Camera Module 3)
- Fuses thermal + RGB detections for crowd estimation

## Hardware Requirements
### Node
- Heltec WiFi LoRa 32 V3 (ESP32‑S3)
- MLX90640 (32×24, I²C)
- Optional: LoRa antenna
### Visualizer
- Raspberry Pi 4B (4GB or 8GB recommended)
- Raspberry Pi Camera Module 3
- MicroSD card (32GB+)
- USB‑C power supply

## Software Requirements
### Raspberry Pi (Python)
For OpenCV, Raspberry Pi OS often requires an older version:
```
pip install  numpy==2.2.4 opencv-python==4.13.0
```

### YOLO26n (ONNX Export)
YOLO26n should be exported to ONNX for faster CPU processing. Export in another device in case of compatibility issues:
```
yolo export model=yolo26n.pt format=onnx imgsz=320 dynamic=False
```

Copy the resulting .onnx file into the Pi.

### Arduino/ESP32
Install via Arduino IDE or PlatformIO:
- Heltec ESP32 board support
- Wire library (built‑in)
- No external MLX90640 library needed (included in repo)

## Getting Started
### 1. Flash the Node
Open /node/ in Arduino IDE or PlatformIO.
Configure pins + I²C speed in config.h:
```cpp
const uint32_t MLX_I2C_SPEED = 1000000; // 1 MHz
const uint8_t MLX90640_ADDRESS = 0x33;
```
Upload to Heltec LoRa V3.
### 2. Prepare the Raspberry Pi
Clone the repo:
```
git clone https://github.com/<yourname>/crowdaware-offline.git
cd crowdaware-offline/python_parser
```
Install dependencies (see above).
Place your exported YOLO26n .onnx file in the folder.
### 3. Run the Visualizer
```
python3 parser_pi.py
```


This will:
- Open the RPi Camera Module 3
- Load YOLO26n (ONNX)
- Parse incoming thermal frames
- Display fused detections

## How It Works
### Thermal Pipeline (ESP32)
- MLX90640 raw frame → temperature map
- Fixed‑point Gaussian smoothing
- Background model (first 25 frames)
- Foreground mask → connected components (watershed algorithm)
- Person candidates filtered by area
### RGB Pipeline (Raspberry Pi)
- YOLO26n ONNX inference
- Bounding box filtering
- Optional fusion with thermal detections

## Troubleshooting
- **MLX90640 not detected**
Check wiring + I²C pull‑ups. The node prints debug messages if SERIAL_OUTPUT_MODE == 0.
- **OpenCV import fails on Raspberry Pi**
Use an older version of OpenCV.
- **Thermal frames look noisy**
Ensure the sensor's power supply is properly decoupled; MLX90640 is sensitive to ambient changes.

### License
MIT License for all original code.
MLX90640 API is © Melexis N.V. under Apache 2.0.
