print("\n🔍 Checking if packages are already installed...")
print("=" * 60)

import vllm
print(f"✅ vllm: {vllm.__version__} (installed)")
# except ImportError:
#     print(f"❌ vllm: not installed")

try:
    import flash_attn
    print(f"✅ flash_attn: {flash_attn.__version__} (installed)")
except ImportError:
    print(f"❌ flash_attn: not installed")

try:
    import transformers
    print(f"✅ transformers: {transformers.__version__} (installed)")
except ImportError:
    print(f"❌ transformers: not installed")

print("\n" + "=" * 60)