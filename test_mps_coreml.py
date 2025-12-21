#!/usr/bin/env python3
"""
Test script to verify MPS (Metal Performance Shaders) and CoreML support
"""

import torch
import torch.nn as nn
import coremltools as ct

def test_mps():
    """Test MPS availability and basic operations"""
    print("=" * 60)
    print("Testing MPS (Metal Performance Shaders) Support")
    print("=" * 60)
    
    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    
    print(f"MPS Available: {mps_available}")
    print(f"MPS Built: {mps_built}")
    
    if mps_available:
        # Set device
        device = torch.device("mps")
        print(f"Using device: {device}")
        
        # Create a simple tensor and perform operations
        print("\nTesting tensor operations on MPS...")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print(f"✓ Matrix multiplication successful: {z.shape}")
        
        # Test a simple neural network
        print("\nTesting neural network on MPS...")
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        ).to(device)
        
        input_tensor = torch.randn(32, 100, device=device)
        output = model(input_tensor)
        print(f"✓ Forward pass successful: {output.shape}")
        
        print("\n✅ MPS is working correctly!")
    else:
        print("⚠️  MPS is not available on this system")
        print("   This is normal if you're not on Apple Silicon (M1/M2/M3)")
    
    return mps_available


def test_coreml_conversion():
    """Test CoreML model conversion"""
    print("\n" + "=" * 60)
    print("Testing CoreML Conversion")
    print("=" * 60)
    
    try:
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(5, 1)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
        
        model = SimpleModel()
        model.eval()
        
        # Create example input
        example_input = torch.randn(1, 10)
        
        # Trace the model
        print("Tracing model...")
        traced_model = torch.jit.trace(model, example_input)
        
        # Convert to CoreML
        print("Converting to CoreML...")
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
            outputs=[ct.TensorType(name="output")],
            minimum_deployment_target=ct.target.macOS13
        )
        
        print("✓ CoreML conversion successful!")
        print(f"  Model type: {type(mlmodel)}")
        print(f"  Input description: {mlmodel.input_description}")
        print(f"  Output description: {mlmodel.output_description}")
        
        print("\n✅ CoreML conversion is working correctly!")
        return True
        
    except Exception as e:
        print(f"⚠️  CoreML conversion test failed: {e}")
        return False


if __name__ == "__main__":
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CoreMLTools Version: {ct.__version__}")
    print()
    
    mps_ok = test_mps()
    coreml_ok = test_coreml_conversion()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"MPS Support: {'✅ Working' if mps_ok else '❌ Not Available'}")
    print(f"CoreML Conversion: {'✅ Working' if coreml_ok else '❌ Failed'}")
    print("=" * 60)

