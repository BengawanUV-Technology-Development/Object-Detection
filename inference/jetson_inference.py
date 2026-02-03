"""
YOLOv5 Inference for Jetson Nano
================================
Real-time object detection using ONNX or TensorRT engine.

Supports:
- ONNX Runtime inference
- TensorRT engine inference (faster)
- USB Camera / CSI Camera input
- Video file input

Usage:
    python jetson_inference.py
    python jetson_inference.py --model model.onnx --source 0
    python jetson_inference.py --model model.engine --source video.mp4
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path

# ============================================
# Model Loading
# ============================================

def load_onnx_model(model_path):
    """Load ONNX model using ONNX Runtime"""
    try:
        import onnxruntime as ort
        
        # Use CUDA provider if available, else CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(model_path), providers=providers)
        
        print(f"✅ Loaded ONNX model: {model_path}")
        print(f"   Providers: {session.get_providers()}")
        
        return session, 'onnx'
    except Exception as e:
        print(f"❌ Failed to load ONNX: {e}")
        return None, None

def load_tensorrt_model(model_path):
    """Load TensorRT engine"""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        logger = trt.Logger(trt.Logger.WARNING)
        
        with open(model_path, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        
        context = engine.create_execution_context()
        
        print(f"✅ Loaded TensorRT engine: {model_path}")
        
        return (engine, context), 'tensorrt'
    except Exception as e:
        print(f"❌ Failed to load TensorRT: {e}")
        return None, None

def load_model(model_path):
    """Auto-detect and load model"""
    model_path = Path(model_path)
    
    if model_path.suffix == '.engine':
        return load_tensorrt_model(model_path)
    elif model_path.suffix == '.onnx':
        return load_onnx_model(model_path)
    else:
        print(f"❌ Unsupported model format: {model_path.suffix}")
        return None, None

# ============================================
# Preprocessing
# ============================================

def preprocess(image, img_size=640):
    """Preprocess image for YOLOv5"""
    # Resize with letterbox
    h, w = image.shape[:2]
    scale = min(img_size / h, img_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to square
    pad_h = (img_size - new_h) // 2
    pad_w = (img_size - new_w) // 2
    
    padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    # Normalize and transpose
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC to CHW
    blob = np.expand_dims(blob, 0)  # Add batch dimension
    
    return blob, (scale, pad_w, pad_h)

# ============================================
# Inference
# ============================================

def inference_onnx(session, blob):
    """Run ONNX inference"""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    return outputs[0]

def inference_tensorrt(model, blob):
    """Run TensorRT inference"""
    import pycuda.driver as cuda
    
    engine, context = model
    
    # Allocate buffers (simplified - real implementation needs proper buffer management)
    # This is a basic example, production code should handle bindings properly
    
    # For now, fallback to ONNX message
    print("⚠️  TensorRT inference not fully implemented in this demo")
    print("   Use ONNX model for testing, or implement full TensorRT pipeline")
    return None

def run_inference(model, model_type, blob):
    """Run inference based on model type"""
    if model_type == 'onnx':
        return inference_onnx(model, blob)
    elif model_type == 'tensorrt':
        return inference_tensorrt(model, blob)
    return None

# ============================================
# Post-processing
# ============================================

def postprocess(outputs, conf_threshold=0.5, iou_threshold=0.45, 
                scale_info=None, orig_shape=None):
    """Post-process YOLOv5 outputs"""
    
    # outputs shape: [1, num_detections, 85] for COCO (80 classes + 5)
    # [x, y, w, h, conf, class_probs...]
    
    predictions = outputs[0]
    
    # Filter by confidence
    mask = predictions[:, 4] > conf_threshold
    predictions = predictions[mask]
    
    if len(predictions) == 0:
        return []
    
    # Get class IDs and scores
    class_ids = np.argmax(predictions[:, 5:], axis=1)
    scores = predictions[:, 4] * predictions[np.arange(len(predictions)), 5 + class_ids]
    
    # Convert xywh to xyxy
    boxes = predictions[:, :4].copy()
    boxes[:, 0] = predictions[:, 0] - predictions[:, 2] / 2  # x1
    boxes[:, 1] = predictions[:, 1] - predictions[:, 3] / 2  # y1
    boxes[:, 2] = predictions[:, 0] + predictions[:, 2] / 2  # x2
    boxes[:, 3] = predictions[:, 1] + predictions[:, 3] / 2  # y2
    
    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), 
        conf_threshold, iou_threshold
    )
    
    results = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            box = boxes[i]
            
            # Scale back to original image
            if scale_info and orig_shape:
                scale, pad_w, pad_h = scale_info
                box[0] = (box[0] - pad_w) / scale
                box[1] = (box[1] - pad_h) / scale
                box[2] = (box[2] - pad_w) / scale
                box[3] = (box[3] - pad_h) / scale
            
            results.append({
                'box': box.astype(int).tolist(),
                'class_id': int(class_ids[i]),
                'score': float(scores[i])
            })
    
    return results

# ============================================
# Visualization
# ============================================

# Default class names (COCO dataset - adjust for your custom classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def draw_detections(image, detections, class_names=None):
    """Draw bounding boxes on image"""
    if class_names is None:
        class_names = COCO_CLASSES
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        class_id = det['class_id']
        score = det['score']
        
        # Get class name
        if class_id < len(class_names):
            label = f"{class_names[class_id]}: {score:.2f}"
        else:
            label = f"Class {class_id}: {score:.2f}"
        
        # Draw box
        color = (0, 255, 0)  # Green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1 + 5, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(description='YOLOv5 Inference for Jetson Nano')
    parser.add_argument('--model', type=str, default='models/model.onnx',
                        help='Path to ONNX or TensorRT engine model')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: camera index (0, 1) or video file path')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Inference image size')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--save', action='store_true',
                        help='Save output video')
    parser.add_argument('--show', action='store_true', default=True,
                        help='Show output window')
    args = parser.parse_args()
    
    # Load model
    model, model_type = load_model(args.model)
    if model is None:
        print("❌ Failed to load model")
        return
    
    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video source: {args.source}")
        return
    
    print(f"📷 Video source: {args.source}")
    print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"\n🚀 Starting inference... (Press 'q' to quit)")
    
    # Video writer for saving
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = 'outputs/output.mp4'
        writer = cv2.VideoWriter(out_path, fourcc, 30, 
                                  (int(cap.get(3)), int(cap.get(4))))
        print(f"💾 Saving to: {out_path}")
    
    fps_history = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Preprocess
            blob, scale_info = preprocess(frame, args.imgsz)
            
            # Inference
            outputs = run_inference(model, model_type, blob)
            
            if outputs is not None:
                # Postprocess
                detections = postprocess(
                    outputs, args.conf, args.iou,
                    scale_info, frame.shape[:2]
                )
                
                # Draw detections
                frame = draw_detections(frame, detections)
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # Draw FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save frame
            if writer:
                writer.write(frame)
            
            # Show frame
            if args.show:
                cv2.imshow('YOLOv5 Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Average FPS: {sum(fps_history)/len(fps_history):.1f}" if fps_history else "")

if __name__ == '__main__':
    main()
