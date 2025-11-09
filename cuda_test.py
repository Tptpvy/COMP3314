# test_cuda.py
import torch

print("=== CUDA Availability Test ===")

# Basic CUDA check
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Print GPU information
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Test tensor operations on GPU
    print("\n=== Testing GPU Operations ===")
    device = torch.device('cuda')
    
    # Create tensors
    a = torch.randn(1000, 1000).to(device)
    b = torch.randn(1000, 1000).to(device)
    
    # Perform operation
    c = torch.matmul(a, b)
    
    print(f"Tensor a device: {a.device}")
    print(f"Tensor b device: {b.device}")
    print(f"Result tensor c device: {c.device}")
    print(f"Result shape: {c.shape}")
    print("✓ GPU operations working correctly!")
    
else:
    print("CUDA is not available. Possible reasons:")
    print("1. PyTorch was installed without CUDA support")
    print("2. No NVIDIA GPU detected")
    print("3. NVIDIA drivers are not installed")
    print("4. CUDA toolkit is not installed")

# Check PyTorch build information
print(f"\n=== PyTorch Build Information ===")
print(f"PyTorch version: {torch.__version__}")
print(f"Build with CUDA: {torch.version.cuda is not None}")

# Test if we can move model to GPU
print(f"\n=== Model GPU Test ===")
try:
    model = torch.nn.Linear(10, 5)
    model = model.to(device)
    print(f"Model successfully moved to: {next(model.parameters()).device}")
except Exception as e:
    print(f"Failed to move model to GPU: {e}")