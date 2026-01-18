from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        
    else:
        print("akdfjalkjfdl")