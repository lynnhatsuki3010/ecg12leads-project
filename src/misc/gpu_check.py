# misc/gpu_check.py
import torch

def check_gpu():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ Không phát hiện GPU.")

if __name__ == "__main__":
    check_gpu()
