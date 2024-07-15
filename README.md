# miniGPT
A framework for training GPT models with mixed precision, gradient accumulation, and other features.

## Features

- **Mixed Precision Training**: Leverage FP16 and BFloat16 for faster training and reduced memory usage.
- **Gradient Accumulation**: Simulate larger batch sizes without running out of GPU memory.
- **Optimized Data Loading**: Efficient data loading using multiple workers to fully utilize CPU resources.
- **Checkpoint Management**: Save and load model and optimizer states for easy training resumption.
- **Flexible Configuration**: Easily adjust batch size, accumulation steps, and other parameters to suit your training setup.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Transformers
- tqdm

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/SimonVutov/miniGPT.git
    ```
2.1 downloading the following:
CUDA Toolkit 12.5.0 (May 2024)
cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x

2.2 Install the required packages:
    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
    pip install datasets transformers torch tqdm
    ```

### Usage

1. **Training**: Run the training script to start training the GPT model.
    ```bash
    python main.py
    ```

2. **Generate Text**: Use the `generate_text` function to generate text using the trained model.
    ```python
    from main import generate_text
    generate_text("Your input text here")
    ```

### Example

An example of training output and generated text:

Device:  cuda

Model and optimizer loaded from checkpoint 'gpt2_epoch_1.pt'

Epoch 1, Batch 200, Loss: 4.9928, Tokens/sec: 18374.01, Time Elapsed: 55.16 sec

Epoch 1, Batch 400, Loss: 4.4935, Tokens/sec: 18840.11, Time Elapsed: 74.05 sec

Other
Create torch virtual environment: python -m venv torch-env
Activate it: torch-env\Scripts\activate