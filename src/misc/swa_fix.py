"""
Fix lỗi load SWA model với prefix "module."
"""

import torch
from collections import OrderedDict

def load_swa_model(model_path, model, device='cuda'):
    """
    Load SWA model và tự động fix prefix "module."
    
    Args:
        model_path: Đường dẫn đến file .pth
        model: Model architecture đã khởi tạo
        device: cuda hoặc cpu
    
    Returns:
        model: Model đã load weights
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Nếu checkpoint là dict (có thể có metadata)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove "module." prefix nếu có
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Bỏ qua key "n_averaged" (từ SWA)
        if k == 'n_averaged':
            continue
        
        # Remove "module." prefix
        if k.startswith('module.'):
            name = k[7:]  # remove 'module.' (7 characters)
        else:
            name = k
        
        new_state_dict[name] = v
    
    # Load vào model
    model.load_state_dict(new_state_dict, strict=True)
    
    return model