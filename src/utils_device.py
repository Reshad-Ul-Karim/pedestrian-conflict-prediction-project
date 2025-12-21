"""
Device utilities for MPS (Metal Performance Shaders) and CoreML support
"""

import torch
import coremltools as ct
from typing import Optional, Union


def get_device() -> torch.device:
    """
    Get the best available device (MPS for Apple Silicon, CUDA for NVIDIA, else CPU)
    
    Returns:
        torch.device: The device to use for computations
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def is_mps_available() -> bool:
    """Check if MPS is available"""
    return torch.backends.mps.is_available()


def is_cuda_available() -> bool:
    """Check if CUDA is available"""
    return torch.cuda.is_available()


def convert_to_coreml(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_path: str,
    input_name: str = "input",
    output_name: str = "output",
    minimum_deployment_target: Optional[ct.target] = None
) -> ct.models.MLModel:
    """
    Convert a PyTorch model to CoreML format
    
    Args:
        model: PyTorch model to convert
        example_input: Example input tensor for tracing
        output_path: Path to save the CoreML model (.mlmodel)
        input_name: Name for the input tensor
        output_name: Name for the output tensor
        minimum_deployment_target: Minimum macOS/iOS version (default: macOS 13)
    
    Returns:
        CoreML model object
    """
    model.eval()
    
    # Set minimum deployment target if not provided
    if minimum_deployment_target is None:
        minimum_deployment_target = ct.target.macOS13
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name=input_name, shape=example_input.shape)],
        outputs=[ct.TensorType(name=output_name)],
        minimum_deployment_target=minimum_deployment_target
    )
    
    # Save the model
    mlmodel.save(output_path)
    print(f"✓ CoreML model saved to: {output_path}")
    
    return mlmodel


def optimize_for_mps(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize a model for MPS execution
    
    Args:
        model: PyTorch model to optimize
    
    Returns:
        Optimized model (moved to MPS device)
    """
    device = get_device()
    if device.type == "mps":
        model = model.to(device)
        # Enable optimizations for MPS
        torch.backends.mps.empty_cache()  # Clear cache if needed
    return model


def print_device_info():
    """Print information about available devices"""
    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS Available: {is_mps_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")
    print(f"CUDA Available: {is_cuda_available()}")
    if is_cuda_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Recommended Device: {get_device()}")
    print("=" * 60)

