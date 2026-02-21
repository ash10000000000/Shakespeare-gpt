# Shakespeare-GPT: Transformer from Scratch

A character-level Generative Pre-trained Transformer (GPT) trained on the works of William Shakespeare. This project explores the transition from a simple Bigram model to a full-scale Transformer architecture, specifically optimized for NVIDIA RTX 50-series (Blackwell) hardware.

## Hardware Optimization
This implementation is specifically tuned for the RTX 5060. Key optimizations include:
- Architecture Support: Compiled using PyTorch Nightly for sm_120 (Blackwell) compatibility.
- Tensor Cores: Enabled float32_matmul_precision('high') to leverage Blackwell Tensor Cores for 2-3x faster training.
- Efficient Inference: Minimal-latency generation scripts with character-level tokenization.



## Project Structure
* gpt2shake.py: The core Transformer training script including Multi-Head Attention and Feed-Forward blocks.
* bigrammodel.py: A baseline Bigram language model for performance comparison.
* chat.py: An interactive inference script with unique session logging and timestamping.
* model_weights.pth: The pre-trained weights (approx. 10.79M parameters).
* input.txt: The "Tiny Shakespeare" dataset.

## Installation and Setup

### Prerequisites
- Python 3.12 (Recommended for PyTorch compatibility)
- NVIDIA RTX 50-series GPU
- CUDA 12.4+

### Setup Environment
1. Clone the repository:
   git clone https://github.com/ash10000000000/Shakespeare-gpt.git
   cd Shakespeare-gpt

2. Create and activate a virtual environment:
   py -3.12 -m venv venv
   .\venv\Scripts\activate

3. Install dependencies (using the PyTorch Nightly build for Blackwell support):
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   pip install tqdm

## Training Results
The model consists of 10.79 million parameters and was trained for 5,000 iterations.
- Batch Size: 64
- Block Size: 256
- Embedding Dimension: 384
- Attention Heads: 6
- Final Validation Loss: ~1.48



## Interacting with the Model
To talk to the model without retraining, ensure model_weights.pth is in the directory and run:
python chat.py

This will generate a unique log file for your session recorded with precise timestamps.

## License
MIT License - feel free to use and modify for your own projects!
