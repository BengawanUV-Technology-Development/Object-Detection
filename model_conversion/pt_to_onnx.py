"""
YOLOv5 Model Export to ONNX
===========================
Export trained YOLOv5n model to ONNX format for Jetson Nano deployment.

ONNX model will be converted to TensorRT engine on Jetson Nano for
optimal inference performance.

Usage:
    python pt_to_onnx.py
    python pt_to_onnx.py --weights path/to/best.pt
    python pt_to_onnx.py --weights best.pt --imgsz 640
"""

import subprocess
import sys
import os
from pathlib import Path

def find_best_weights():
    """Find the best.pt from training runs"""
    runs_dir = Path('../runs/train')
    
    if not runs_dir.exists():
        print("❌ Folder runs/train tidak ditemukan")
        return None
    
    # Find most recent training run
    exp_dirs = sorted(runs_dir.glob('*/weights/best.pt'), 
                    key=lambda x: x.stat().st_mtime, 
                    reverse=True)
    
    if exp_dirs:
        return exp_dirs[0]
    
    # Check if model.pt exists in current directory
    if Path('model.pt').exists():
        return Path('model.pt')
    
    return None

def export_to_onnx(weights_path, imgsz=640, opset=12):
    """Export YOLOv5 weights to ONNX format"""
    
    yolov5_path = Path('../yolov5')
    export_script = yolov5_path / 'export.py'
    
    if not export_script.exists():
        print("❌ YOLOv5 repository tidak ditemukan")
        print("   Jalankan train_yolov5.py terlebih dahulu untuk clone repo")
        return 1
    
    output_path = Path(weights_path).with_suffix('.onnx')
    
    cmd = [
        sys.executable, str(export_script),
        '--weights', str(weights_path),
        '--imgsz', str(imgsz),
        '--opset', str(opset),  # ONNX opset version (12 for TensorRT compatibility)
        '--include', 'onnx',
        '--simplify',  # Simplify ONNX model
    ]
    
    print(f"🔄 Exporting to ONNX...")
    print(f"   Input: {weights_path}")
    print(f"   Output: {output_path}")
    print(f"   Image size: {imgsz}")
    print(f"   ONNX opset: {opset}")
    print()
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n✅ Export berhasil!")
        print(f"📁 ONNX model: {output_path}")
        print(f"\n📋 Langkah selanjutnya di Jetson Nano:")
        print(f"   1. Transfer {output_path.name} ke Jetson Nano")
        print(f"   2. Convert ke TensorRT:")
        print(f"      /usr/src/tensorrt/bin/trtexec --onnx={output_path.name} --saveEngine=model.engine --fp16")
    else:
        print(f"\n❌ Export gagal")
    
    return result.returncode

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Export YOLOv5 to ONNX')
    parser.add_argument('--weights', type=str, default=None,
                        help='path to weights file (default: auto-detect)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='image size')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Find weights file
    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = find_best_weights()
    
    if weights_path is None or not weights_path.exists():
        print("❌ Tidak dapat menemukan file weights (.pt)")
        print("   Gunakan: python pt_to_onnx.py --weights path/to/best.pt")
        sys.exit(1)
    
    print(f"📦 Found weights: {weights_path}")
    sys.exit(export_to_onnx(weights_path, args.imgsz, args.opset))