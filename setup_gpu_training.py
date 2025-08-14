"""
Setup script for GPU-accelerated DistilBERT fake news detection
Installs PyTorch with CUDA support for NVIDIA GTX 1050
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    if description:
        print(f"📦 {description}")
    print(f"🔧 Running: {command}")
    print('='*60)
    
    try:
        # Use timeout to prevent hanging
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True, timeout=300)
        print("✅ Success!")
        if result.stdout and len(result.stdout.strip()) > 0:
            # Show relevant output (last 1000 chars)
            output = result.stdout.strip()
            if len(output) > 1000:
                print("Output (last 1000 chars):", "..." + output[-1000:])
            else:
                print("Output:", output)
        return True
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 5 minutes")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Command failed with exit code {e.returncode}")
        if e.stdout and len(e.stdout.strip()) > 0:
            print("STDOUT:", e.stdout.strip()[-500:])  # Last 500 chars
        if e.stderr and len(e.stderr.strip()) > 0:
            print("STDERR:", e.stderr.strip()[-500:])  # Last 500 chars
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_gpu():
    """Check for NVIDIA GPU and CUDA"""
    print("\n🔍 Checking GPU and CUDA availability...")
    
    # Check for nvidia-smi
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected!")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GTX' in line or 'RTX' in line or 'GeForce' in line:
                    print(f"   GPU: {line.strip()}")
                    break
            return True
        else:
            print("❌ nvidia-smi not found")
            return False
    except:
        print("❌ Unable to detect NVIDIA GPU")
        return False

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    print("\n🚀 Installing PyTorch with CUDA support...")
    
    # Try multiple CUDA versions and installation methods
    install_commands = [
        # CUDA 12.1 (latest stable)
        ("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121", "CUDA 12.1"),
        # CUDA 11.8 (older but more compatible)
        ("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", "CUDA 11.8"),
        # Force reinstall with no cache
        ("pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", "CUDA 11.8 (no cache)"),
        # CPU fallback
        ("pip install torch torchvision torchaudio", "CPU only"),
    ]
    
    for command, description in install_commands:
        print(f"\n🔄 Trying {description}...")
        if run_command(command, f"Installing PyTorch with {description}"):
            print(f"✅ PyTorch with {description} installed successfully!")
            return True
        else:
            print(f"❌ Failed to install PyTorch with {description}")
            if "CPU only" in description:
                break
            print("   Trying next option...")
    
    print("❌ All PyTorch installation attempts failed")
    return False

def install_transformers():
    """Install Transformers and related packages"""
    packages = [
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "tokenizers",
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"❌ Failed to install {package}")
            return False
    
    print("✅ All transformer packages installed!")
    return True

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    test_script = '''
import torch
import transformers

print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
    
    # Test basic GPU operations
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.matmul(x, x)
        print("✅ GPU operations working!")
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
else:
    print("⚠️ CUDA not available - will use CPU mode")

# Test DistilBERT loading
try:
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    print("✅ DistilBERT model loading successful!")
except Exception as e:
    print(f"❌ DistilBERT loading failed: {e}")

print("\\n🎉 Installation test complete!")
'''
    
    try:
        with open("test_setup.py", "w") as f:
            f.write(test_script)
        
        result = subprocess.run([sys.executable, "test_setup.py"], 
                              capture_output=True, text=True, timeout=60)
        
        print("Test output:")
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        # Clean up
        os.remove("test_setup.py")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 GPU-Accelerated DistilBERT Setup for GTX 1050")
    print("=" * 60)
    print("This script will:")
    print("1. Check for NVIDIA GPU and CUDA")
    print("2. Install PyTorch with CUDA support")
    print("3. Install Transformers and dependencies")
    print("4. Test the installation")
    print("5. Provide training instructions")
    print()
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python {python_version.major}.{python_version.minor} is too old.")
        print("   Please use Python 3.8 or newer")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check platform
    if platform.system() != "Windows":
        print("⚠️ This script is optimized for Windows with NVIDIA GPU")
        print("   It should work on other platforms but may need adjustments")
    
    # Step 1: Check GPU
    gpu_available = check_gpu()
    
    # Step 2: Install PyTorch
    if not install_pytorch_cuda():
        print("❌ PyTorch installation failed")
        return False
    
    # Step 3: Install Transformers
    if not install_transformers():
        print("❌ Transformers installation failed")
        return False
    
    # Step 4: Test installation
    if not test_installation():
        print("❌ Installation test failed")
        print("   The packages are installed but may not be working correctly")
    
    # Step 5: Training instructions
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print()
    
    if gpu_available:
        print("🚀 GPU-accelerated training available!")
        print("   Your NVIDIA GPU is ready for DistilBERT training")
    else:
        print("💻 CPU-only mode")
        print("   DistilBERT will run on CPU (slower but still functional)")
    
    print("\nNext steps:")
    print("1. Train the models:")
    print("   python train_hybrid_models.py --quick    # Quick training (5-10 min)")
    print("   python train_hybrid_models.py            # Full training (30-60 min)")
    print()
    print("2. Test individual models:")
    print("   python train_hybrid_models.py --distilbert-only --quick")
    print("   python train_hybrid_models.py --lr-only --quick")
    print()
    print("3. Start the web application:")
    print("   python run.py")
    print("   Visit: http://localhost:5000/fake-news/check")
    print()
    print("4. Check model performance:")
    print("   Visit: http://localhost:5000/fake-news/about")
    print()
    
    # GTX 1050 specific advice
    if gpu_available:
        print("💡 GTX 1050 Optimization Tips:")
        print("   • Use --quick mode for initial testing")
        print("   • Batch size is automatically optimized for 2GB VRAM")
        print("   • Training will use gradient accumulation for efficiency")
        print("   • If you get CUDA out of memory errors, restart and try again")
        print()
    
    print("📚 For more information, check the documentation in:")
    print("   docs/database-backup-system.md")
    print("   docs/rss-feed-crud.md")
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)
