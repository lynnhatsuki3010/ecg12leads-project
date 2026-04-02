# training/augmentations.py
import numpy as np

def jitter(x, sigma=0.01):
    return x + np.random.normal(0, sigma, size=x.shape)

def scaling(x, sigma=0.1):
    factor = np.random.normal(1.0, sigma, (x.shape[0],1))
    return x * factor

def time_shift(x, max_shift=0.05):
    # shift fraction of length
    L = x.shape[1]
    shift = np.random.randint(-int(max_shift*L), int(max_shift*L))
    if shift > 0:
        return np.concatenate([x[:,shift:], np.zeros((x.shape[0], shift))], axis=1)
    elif shift < 0:
        s = -shift
        return np.concatenate([np.zeros((x.shape[0], s)), x[:,:-s]], axis=1)
    else:
        return x

def random_crop_or_pad(x, target_len):
    L = x.shape[1]
    if L > target_len:
        start = np.random.randint(0, L-target_len)
        return x[:, start:start+target_len]
    elif L < target_len:
        pad = target_len - L
        left = pad // 2
        right = pad - left
        return np.pad(x, ((0,0),(left,right)), mode='constant')
    else:
        return x






# training/augmentations.py
# import numpy as np

# def jitter(x, sigma=0.01):
#     """Add Gaussian noise"""
#     noise = np.random.normal(0, sigma, x.shape)
#     return x + noise.astype(np.float32)


# def scaling(x, sigma=0.1):
#     """Scale signal randomly"""
#     scale = np.random.normal(1.0, sigma)
#     return x * scale


# def time_shift(x, max_shift=0.05):
#     """
#     Randomly shift signal in time
#     Args:
#         x: (seq_len, n_leads) shape
#         max_shift: fraction of sequence length to shift
#     """
#     seq_len = x.shape[0]
#     max_shift_samples = int(max_shift * seq_len)
    
#     if max_shift_samples == 0:
#         return x
    
#     shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
    
#     if shift == 0:
#         return x
#     elif shift > 0:
#         # Shift right: pad left, truncate right
#         return np.concatenate([np.zeros((shift, x.shape[1])), x[:-shift]], axis=0)
#     else:
#         # Shift left: truncate left, pad right
#         return np.concatenate([x[-shift:], np.zeros((-shift, x.shape[1]))], axis=0)


# def rotation(x, sigma=0.2):
#     """Random rotation/flip"""
#     flip = np.random.choice([-1, 1], size=(x.shape[1],))
#     rotate_axis = np.arange(x.shape[1])
#     np.random.shuffle(rotate_axis)
#     return flip * x[:, rotate_axis]


# def permutation(x, max_segments=5, seg_mode="equal"):
#     """Randomly permute segments"""
#     seq_len = x.shape[0]
#     num_segs = np.random.randint(1, max_segments + 1)
    
#     if seg_mode == "equal":
#         split_points = np.linspace(0, seq_len, num_segs + 1).astype(int)
#     else:
#         split_points = np.sort(np.random.choice(seq_len, num_segs - 1, replace=False))
#         split_points = np.concatenate([[0], split_points, [seq_len]])
    
#     segments = [x[split_points[i]:split_points[i+1]] for i in range(num_segs)]
#     np.random.shuffle(segments)
    
#     return np.concatenate(segments, axis=0)