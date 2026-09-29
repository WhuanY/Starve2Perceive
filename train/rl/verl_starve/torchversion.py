import torch
import sys

print("=" * 60)
print("PyTorch & CUDA Environment Check")
print("=" * 60)

# 1. PyTorch版本和CUDA版本
print(f"\n📦 PyTorch Version: {torch.__version__}")
print(f"🔧 CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"🎮 CUDA Version (PyTorch compiled with): {torch.version.cuda}")
    print(f"🎮 cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"🔢 Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("⚠️  CUDA not available!")

# 2. 检查C++ ABI (关键！决定用哪个flash-attn版本)
print(f"\n🔨 C++ ABI (CXX11_ABI): {torch._C._GLIBCXX_USE_CXX11_ABI}")
print(f"   (True=新ABI, False=旧ABI)")

# 3. Python版本
print(f"\n🐍 Python Version: {sys.version}")