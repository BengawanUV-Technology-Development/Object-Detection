from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        device = 'cuda'
        
    else:
        print("gak kedetek gpu")
        device = 'cpu'
    
    model = YOLO('yolo11n.pt')

    results = model.train(
        data='datasets/data_1/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        name='data_1',
        workers=4,
        patience=10,
        save=True
    )

    metrics = model.val()
    print(f"rampung, map50 : {metrics.box.map50}")
