"""
Device management utilities for GPU-accelerated SDT execution.

Provides automatic device selection (MPS on Apple Silicon, CUDA on NVIDIA,
CPU fallback) with graceful degradation when PyTorch is not available.
"""

import os
import warnings
from typing import Optional

# Lazy import of torch to keep CPU import path fast
_torch_available = False
_torch_dtype = None
_device = None


def _check_torch() -> bool:
    """Check if PyTorch is available and return True if so."""
    global _torch_available
    if not _torch_available:
        try:
            import torch
            _torch_available = True
            _torch_dtype = torch.float32
        except ImportError:
            _torch_available = False
    return _torch_available


def get_device() -> Optional[str]:
    """Get the best available device for computation.
    
    Priority: MPS (Apple Silicon GPU) > CUDA (NVIDIA GPU) > CPU
    
    Returns:
        Device string ('mps', 'cuda', or 'cpu') or None if torch not available.
    """
    global _device
    if _device is not None:
        return _device
    
    if not _check_torch():
        _device = None  # torch not available, fall back to numpy everywhere
        return None
    
    import torch
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _device = 'mps'
    elif torch.cuda.is_available():
        _device = 'cuda'
    else:
        _device = 'cpu'
    
    return _device


def has_gpu() -> bool:
    """Check if GPU acceleration is available."""
    dev = get_device()
    return dev is not None and dev in ('mps', 'cuda')


def get_torch_dtype():
    """Get the PyTorch dtype (float32) if torch is available."""
    if _check_torch():
        return _torch_dtype
    return None


def to_torch(array, device=None):
    """Convert numpy array to torch tensor on the appropriate device.
    
    Args:
        array: numpy array or sparse matrix to convert
        device: optional device string ('mps', 'cuda', 'cpu')
    
    Returns:
        torch tensor or None if torch not available
    """
    if not _check_torch():
        return None
    
    import torch
    import scipy.sparse as sp
    
    dev = device or get_device()
    
    if sp.issparse(array):
        # Convert sparse matrix to COO format for torch
        if hasattr(array, 'tocoo'):
            coo = array.tocoo()
            indices = torch.from_numpy(
                np.vstack((coo.row, coo.col)).astype(np.int64)
            )
            values = torch.from_numpy(coo.data.astype(np.float32))
            shape = torch.Size(coo.shape)
            return torch.sparse_coo_tensor(indices, values, shape).to(dev)
        else:
            # Dense fallback for other sparse formats
            return torch.from_numpy(array.toarray().astype(np.float32)).to(dev)
    else:
        return torch.from_numpy(array.astype(np.float32)).to(dev)


def from_torch(tensor):
    """Convert torch tensor back to numpy array.
    
    Args:
        tensor: torch tensor or sparse tensor
    
    Returns:
        numpy array or None if tensor is None
    """
    if tensor is None:
        return None
    
    import numpy as np
    import scipy.sparse as sp
    
    if hasattr(tensor, 'to_dense'):
        # Sparse tensor
        dense = tensor.to_dense()
        return dense.cpu().numpy()
    elif tensor.is_sparse:
        # torch.sparse tensor
        return tensor.to_dense().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def maybe_device(array, device=None):
    """Convert to device if GPU available, else return original array.
    
    Used as a wrapper around numpy arrays - if GPU is available,
    converts to the appropriate device. Otherwise returns the original.
    
    Args:
        array: numpy array
        device: optional device string
    
    Returns:
        torch tensor on device or original numpy array
    """
    dev = get_device()
    if dev is None or dev == 'cpu':
        return array
    
    torch_tensor = to_torch(array, device or dev)
    return torch_tensor


def maybe_to_device(tensor, device=None):
    """Move a torch tensor to the target device if needed."""
    if tensor is None:
        return None
    
    dev = device or get_device()
    if dev is None:
        return tensor
    
    if tensor.device.type != (dev if dev else 'cpu'):
        return tensor.to(dev)
    return tensor


def matrix_multiply(A, B, device=None):
    """Matrix multiplication that works with both numpy and torch.
    
    Args:
        A: numpy array or torch tensor or sparse tensor
        B: numpy array or torch tensor or sparse tensor
        device: optional target device
    
    Returns:
        Result of matrix multiplication (same type as inputs)
    """
    dev = get_device()
    
    if dev is not None and dev != 'cpu':
        # Use torch for GPU acceleration
        if hasattr(A, 'to_dense') or hasattr(A, 'to'):
            A_t = A if hasattr(A, 'to_dense') else to_torch(A, dev)
        else:
            A_t = to_torch(A, dev) if not isinstance(A, torch.Tensor) else A
        
        if hasattr(B, 'to_dense') or hasattr(B, 'to'):
            B_t = B if hasattr(B, 'to_dense') else to_torch(B, dev)
        else:
            B_t = to_torch(B, dev) if not isinstance(B, torch.Tensor) else B
        
        return torch.matmul(A_t, B_t)
    else:
        # Use numpy/scipy
        import numpy as np
        import scipy.sparse as sp
        
        if sp.issparse(A) or sp.issparse(B):
            return A @ B
        return np.dot(A, B)


# Track GPU usage stats
_gpu_stats = {
    'mps_peak_memory': 0.0,
    'cuda_peak_memory': 0.0,
}


def get_memory_usage(device=None):
    """Get current memory usage on the target device.
    
    Args:
        device: optional device string
    
    Returns:
        Dictionary with memory stats in MB
    """
    import torch
    
    dev = device or get_device()
    stats = {}
    
    if dev == 'mps' and hasattr(torch.backends.mps, 'peak_allocated_memory'):
        stats['mps_allocated'] = torch.backends.mps.peak_allocated_memory() / (1024 ** 2)
        stats['mps_peak'] = torch.backends.mps.peak_reserved_memory() / (1024 ** 2)
    elif dev == 'cuda' and torch.cuda.is_available():
        stats['cuda_allocated'] = torch.cuda.memory_allocated() / (1024 ** 2)
        stats['cuda_reserved'] = torch.cuda.memory_reserved() / (1024 ** 2)
    
    return stats


def clear_gpu_cache(device=None):
    """Clear GPU memory cache.
    
    Args:
        device: optional device string
    """
    if not _check_torch():
        return
    
    import torch
    
    dev = device or get_device()
    if dev == 'mps':
        # MPS doesn't have explicit cache clearing
        pass
    elif dev == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()


# Import numpy at module level for sparse handling
import numpy as np


def log_gpu_telemetry(phase: str = ""):
    """Print GPU telemetry for a phase.
    
    Args:
        phase: phase name for logging (e.g., 'Phase 1')
    """
    if not _check_torch():
        print(f"[GPU] {phase}: torch not available — using numpy/scipy")
        return
    
    import torch
    dev = get_device()
    if dev is None:
        print(f"[GPU] {phase}: no GPU detected — using CPU/numpy")
        return
    
    stats = get_memory_usage(dev)
    
    if dev == 'mps':
        peak_mb = stats.get('mps_allocated', 0)
        print(f"[GPU] {phase}: MPS active — peak memory: {peak_mb:.1f} MB")
    elif dev == 'cuda':
        peak_mb = stats.get('cuda_allocated', 0)
        reserved_mb = stats.get('cuda_reserved', 0)
        print(f"[GPU] {phase}: CUDA active — peak: {peak_mb:.1f} MB, reserved: {reserved_mb:.1f} MB")
    else:
        print(f"[GPU] {phase}: {dev} — using numpy/scipy (no GPU acceleration)")
