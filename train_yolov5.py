"""
YOLOv5n Training Script for Jetson Nano Deployment
===================================================
Training dilakukan di laptop, hasil model di-deploy ke Jetson Nano.

Model: YOLOv5n (nano) - paling ringan, cocok untuk Jetson Nano
Output: runs/train/exp/weights/best.pt

Usage:
    python train_yolov5.py
    python train_yolov5.py --epochs 100 --batch 32
    python train_yolov5.py --data datasets/data_1/data.yaml --epochs 50
"""

import subprocess
import sys
import os
from pathlib import Path

# Configuration
DEFAULT_CONFIG = {
    'data': 'datasets/data_1/data.yaml',
    'epochs': 50,
    'imgsz': 640,
    'batch': 16,
    'name': 'yolov5n_jetson',
    'workers': 4,
    'patience': 10,
}

def check_yolov5_repo():
    """Clone YOLOv5 repository if not exists"""
    yolov5_path = Path('yolov5')
    
    if not yolov5_path.exists():
        print("📥 Cloning YOLOv5 repository...")
        subprocess.run([
            'git', 'clone', 
            'https://github.com/ultralytics/yolov5.git',
            str(yolov5_path)
        ], check=True)
        
        # Install YOLOv5 requirements
        print("📦 Installing YOLOv5 dependencies...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            '-r', str(yolov5_path / 'requirements.txt')
        ], check=True)
    else:
        print("✅ YOLOv5 repository already exists")
    
    return yolov5_path

def train_yolov5(config):
    """Run YOLOv5 training"""
    import torch
    
    # Check GPU
    print(f"🔧 PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        device = '0'  # Use first GPU
    else:
        print("⚠️  GPU tidak terdeteksi, menggunakan CPU")
        device = 'cpu'
    
    # Ensure YOLOv5 repo exists
    yolov5_path = check_yolov5_repo()
    
    # Build training command
    train_script = yolov5_path / 'train.py'
    
    cmd = [
        sys.executable, str(train_script),
        '--weights', 'yolov5n.pt',  # YOLOv5 nano - lightest model
        '--data', config['data'],
        '--epochs', str(config['epochs']),
        '--imgsz', str(config['imgsz']),
        '--batch-size', str(config['batch']),
        '--device', device,
        '--name', config['name'],
        '--workers', str(config['workers']),
        '--patience', str(config['patience']),
        '--exist-ok',
        '--project', 'runs/train'
    ]
    
    print(f"\n🚀 Starting YOLOv5n training...")
    print(f"   Data: {config['data']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch: {config['batch']}")
    print(f"   Image size: {config['imgsz']}")
    print()
    
    # Run training
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        weights_path = Path('runs/train') / config['name'] / 'weights'
        print(f"\n✅ Training selesai!")
        print(f"📁 Model tersimpan di: {weights_path}")
        print(f"\n📋 Langkah selanjutnya:")
        print(f"   1. Export ke ONNX: python model_conversion/pt_to_onnx.py")
        print(f"   2. Transfer ke Jetson Nano")
        print(f"   3. Convert ke TensorRT di Jetson")
    else:
        print(f"\n❌ Training gagal dengan error code: {result.returncode}")
    
    return result.returncode

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv5n Training for Jetson Nano')
    parser.add_argument('--data', type=str, default=DEFAULT_CONFIG['data'],
                        help='path to data.yaml')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help='number of epochs')
    parser.add_argument('--batch', type=int, default=DEFAULT_CONFIG['batch'],
                        help='batch size')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_CONFIG['imgsz'],
                        help='image size')
    parser.add_argument('--name', type=str, default=DEFAULT_CONFIG['name'],
                        help='experiment name')
    parser.add_argument('--workers', type=int, default=DEFAULT_CONFIG['workers'],
                        help='number of workers')
    parser.add_argument('--patience', type=int, default=DEFAULT_CONFIG['patience'],
                        help='early stopping patience')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    config = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'name': args.name,
        'workers': args.workers,
        'patience': args.patience,
    }
    
    sys.exit(train_yolov5(config))
