# COMP3314 Group Project

### Basic Info:
Paper: Attention Is All You Need (https://arxiv.org/abs/1706.03762)
Requirements: Git, Python 3.12.10
Dataset: Multi30k English-to-German (https://github.com/neychev/small_DL_repo/tree/master/datasets/Multi30k) 
         29000 train samples, 1000 test samples

### How to use:
1. cd path\to\directory

2. git clone https://github.com/Tptpvy/COMP3314.git

3. \[Optional\]: Create virtual environment (python3 -m venv venv -> .\venv\Scripts\activate)

4. Execute setup script (setup.bat for Windows, bash setup.sh for Linux/Mac)

5. python3 train_multi30k.py (train & test transformer on Multi30k dataset)

6. Testing results are saved in ./results

### Model Spec:
##### Architecture Details
Model Type: Transformer Sequence-to-Sequence
Implementation: PyTorch
Architecture Style: Encoder-Decoder with Attention
##### Core Dimensions
Embedding Dimension (d_model): 512
Feed-Forward Dimension (d_ff): 2048
Number of Layers: 6 encoder layers, 6 decoder layers
Attention Heads: 8
Dropout Rate: 0.1
Maximum Sequence Length: 100 tokens
##### Parameter Count
Total Parameters: ~54 million
Vocabulary Sizes:
Source (German): ~7,853 tokens
Target (English): ~5,893 tokens
##### Training Specifications
Optimizer: Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹)
Learning Rate Schedule: Warmup + Inverse Square Root Decay
Label Smoothing: 0.1
Gradient Clipping: 1.0
Batch Size: 128
##### Key Features
1. Positional Encoding
Sinusoidal Positional Encoding (non-learned)
Fixed patterns using sine and cosine functions
Allows model to understand token positions

2. Multi-Head Attention
Self-attention in encoder and decoder
Cross-attention in decoder (encoder-decoder attention)
8 parallel attention heads
Scaled dot-product attention

3. Feed-Forward Networks
GELU Activation (Gaussian Error Linear Unit)
Two linear transformations with expansion factor 4
Position-wise fully connected layers

4. Normalization and Regularization
Post-Layer Normalization (after residual connections)
Dropout applied to attention and FFN outputs
Residual Connections around each sub-layer

5. Masking Mechanisms
Source Padding Mask: Hides padding tokens in encoder
Target Causal Mask: Prevents attending to future tokens
Target Padding Mask: Hides padding tokens in decoder

### Additional Info: 
##### Experimental scripts:
cuda_test.py (test if cuda is available)
train_wmt14.py (train & test transformer on WMT14 dataset, currently has cuda compatability issues)

##### Hardware Used:
CPU: 13th Gen Intel(R) Core(TM) i7-13700K (3.40 GHz)
GPU: NVIDIA GeForce RTX 4070

##### References:
https://github.com/retrogtx/attention-is-all-you-need/
https://github.com/hyunwoongko/transformer/