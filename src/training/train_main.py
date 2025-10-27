# training/train_main.py
import torch
from training.resnet_attention import ResNet1DAttention
from training.resnet50_attention import ResNet1D50Attention   # <-- thêm dòng này
from training.inception_time1d import InceptionTime1D
from training.train_utils import train_model, evaluate_model


def run_training(
    train_loader=None,
    val_loader=None,
    test_loader=None,
    y_train=None,
    y_test=None,
    device=None,
    model_name="resnet"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Chọn mô hình ===
    model_name = model_name.lower()
    if model_name == "resnet":
        print("Training ResNet1D + Attention")
        model = ResNet1DAttention(num_classes=8)
    elif model_name == "resnet50":
        print("Training ResNet1D-50 + Attention")
        model = ResNet1D50Attention(num_classes=8)
    elif model_name == "inception":
        print("Training InceptionTime1D + Attention")
        model = InceptionTime1D(in_channels=12, num_classes=8)
    else:
        raise ValueError(f"❌ Unknown model name: {model_name}")

    # === Huấn luyện ===
    model = train_model(model, train_loader, val_loader, y_train, device=device)

    # === Đánh giá ===
    print(f"\n=== Đánh giá mô hình {model_name.upper()} ===")
    evaluate_model(model, test_loader, y_test, device=device)
