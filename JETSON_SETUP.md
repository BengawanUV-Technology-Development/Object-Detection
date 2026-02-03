# Jetson Nano Setup Guide

Panduan setup dan deployment Object Detection pada Jetson Nano Developer Kit.

## Prerequisites

### Hardware
- Jetson Nano Developer Kit (4GB recommended)
- MicroSD Card (64GB+ recommended)
- Power Supply (5V 4A barrel jack recommended)
- USB Camera atau CSI Camera (optional)
- Monitor, keyboard, mouse untuk setup awal

### Software
- JetPack 4.6.1 (L4T R32.7.1)
- CUDA 10.2 (pre-installed dengan JetPack)
- TensorRT 8.0.1 (pre-installed dengan JetPack)
- Python 3.6

---

## Quick Start

### 1. Flash JetPack ke SD Card

Download dan flash JetPack image:
```bash
# Download dari: https://developer.nvidia.com/embedded/jetpack
# Gunakan Etcher atau dd untuk flash ke SD card
```

### 2. Setup Awal Jetson

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-opencv libopencv-dev

# Upgrade pip
pip3 install --upgrade pip
```

### 3. Install PyTorch untuk Jetson

PyTorch untuk Jetson harus diinstall dari wheel NVIDIA:

```bash
# Download PyTorch wheel (JetPack 4.6)
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install PyTorch
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install torchvision
pip3 install torchvision==0.11.1
```

---

## 📦 Deployment Workflow

### File yang Dibutuhkan di Jetson Nano

| File | Sumber | Keterangan |
|------|--------|------------|
| `best.onnx` | Dari laptop (hasil export) | **WAJIB** - model untuk inference |
| `inference/jetson_inference.py` | Clone dari repo | Script inference |
| `requirements.txt` | Clone dari repo | Dependencies |
| `Dockerfile.jetson` | Clone dari repo | Opsional (jika pakai Docker) |

> **Note:** File `yolov5/`, `train*.py`, `datasets/` **TIDAK diperlukan** di Jetson.

### Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAPTOP (Training)                         │
├─────────────────────────────────────────────────────────────────┤
│  1. python train_yolov5.py                                      │
│     → Output: runs/train/yolov5n_jetson/weights/best.pt         │
│                                                                  │
│  2. python model_conversion/pt_to_onnx.py                       │
│     → Output: best.onnx                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Transfer via SCP/USB
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      JETSON NANO (Inference)                     │
├─────────────────────────────────────────────────────────────────┤
│  3. Clone repo: git clone <your-repo>                           │
│                                                                  │
│  4. Copy best.onnx ke folder models/                            │
│                                                                  │
│  5. Convert ONNX to TensorRT (opsional, untuk FPS lebih tinggi) │
│     → Output: model.engine                                       │
│                                                                  │
│  6. python3 inference/jetson_inference.py --model models/best.onnx│
│     → Real-time object detection                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: Training di Laptop

```bash
# Clone repo dan install dependencies
git clone <your-repo>
cd Object-Detection
pip install -r requirements.txt

# Training YOLOv5n
python train_yolov5.py --epochs 50 --batch 16

# Hasil training ada di: runs/train/yolov5n_jetson/weights/best.pt
```

### Step 2: Export ke ONNX

```bash
cd model_conversion
python pt_to_onnx.py --weights ../runs/train/yolov5n_jetson/weights/best.pt

# Output: best.onnx
```

### Step 3: Transfer ke Jetson Nano

```bash
# Dari laptop, transfer via SCP
scp model_conversion/best.onnx jetson@<jetson-ip>:/home/jetson/models/
scp -r inference/ jetson@<jetson-ip>:/home/jetson/
```

### Step 4: Convert ke TensorRT (Opsional)

Konversi ke TensorRT memberikan performa ~2-3x lebih cepat:

```bash
# Di Jetson Nano
/usr/src/tensorrt/bin/trtexec \
    --onnx=model.onnx \
    --saveEngine=model.engine \
    --fp16 \
    --workspace=1024
```

### Step 5: Jalankan Inference

```bash
# Menggunakan ONNX
python3 inference/jetson_inference.py --model models/model.onnx --source 0

# Menggunakan TensorRT (lebih cepat)
python3 inference/jetson_inference.py --model models/model.engine --source 0

# Dari video file
python3 inference/jetson_inference.py --model models/model.onnx --source video.mp4
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
# Di Jetson Nano
docker build -f Dockerfile.jetson -t yolov5-jetson .
```

### Run Container

```bash
# Dengan akses kamera
docker run --runtime nvidia --rm -it \
    --device /dev/video0:/dev/video0 \
    -v $(pwd)/models:/app/models \
    yolov5-jetson

# Dengan GPU dan display
docker run --runtime nvidia --rm -it \
    --device /dev/video0:/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/models:/app/models \
    yolov5-jetson
```

---

## ⚡ Performance Optimization

### 1. Maximize Jetson Performance

```bash
# Set power mode ke MAXN (10W)
sudo nvpmodel -m 0

# Enable all CPU cores
sudo jetson_clocks
```

### 2. Reduce Image Size

Untuk inference lebih cepat, gunakan image size lebih kecil:

```bash
python3 inference/jetson_inference.py --model model.onnx --imgsz 416
```

### 3. Expected Performance

| Model | Backend | Image Size | FPS (approx) |
|-------|---------|------------|--------------|
| YOLOv5n | ONNX | 640 | 8-12 FPS |
| YOLOv5n | TensorRT FP16 | 640 | 20-25 FPS |
| YOLOv5n | TensorRT FP16 | 416 | 30-35 FPS |

---

## 🔧 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size dan image size
# Gunakan swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Camera Not Detected

```bash
# Check camera
ls -l /dev/video*

# Test with v4l2
v4l2-ctl --list-devices
```

### TensorRT Conversion Failed

```bash
# Pastikan ONNX opset compatible
# Gunakan opset 12 saat export
python pt_to_onnx.py --opset 12
```

---

## 📚 Resources

- [NVIDIA Jetson Developer](https://developer.nvidia.com/embedded/jetson-nano)
- [YOLOv5 Documentation](https://docs.ultralytics.com/yolov5/)
- [JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
- [NVIDIA L4T PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)
