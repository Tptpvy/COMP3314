# COMP3314 Group Project

### Basic Info:
Paper: Attention Is All You Need (https://arxiv.org/abs/1706.03762) <br>
Requirements: Git, Python 3.12.10 <br>
Dataset: Multi30k English-to-German (29000 train samples, 1000 test samples, https://github.com/neychev/small_DL_repo/tree/master/datasets/Multi30k)

### How to use:
1. cd path\to\directory

2. git clone https://github.com/Tptpvy/COMP3314.git

3. \[Optional\]: Create virtual environment (python3 -m venv venv -> .\venv\Scripts\activate)

4. Execute setup script (setup.bat for Windows, bash setup.sh for Linux/Mac)

5. python3 train_multi30k.py (train & test transformer on Multi30k dataset)

6. Testing results are saved in ./results

### Model Spec:
##### Architecture Details
Model Type: Transformer Sequence-to-Sequence <br>
Implementation: PyTorch <br>
Architecture Style: Encoder-Decoder with Attention <br>
##### Core Dimensions
Embedding Dimension (d_model): 512 <br>
Feed-Forward Dimension (d_ff): 2048 <br>
Number of Layers: 6 encoder layers, 6 decoder layers <br>
Attention Heads: 8 <br>
Dropout Rate: 0.1 <br>
Maximum Sequence Length: 100 tokens <br>
##### Parameter Count
Total Parameters: ~54 million <br>
Vocabulary Sizes: <br>
Source (German): ~7,853 tokens <br>
Target (English): ~5,893 tokens <br>
##### Training Specifications
Optimizer: Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹) <br>
Learning Rate Schedule: Warmup + Inverse Square Root Decay <br>
Label Smoothing: 0.1 <br>
Gradient Clipping: 1.0 <br>
Batch Size: 128 <br>
##### Key Features
1. Positional Encoding
- Sinusoidal Positional Encoding (non-learned)
- Fixed patterns using sine and cosine functions
- Allows model to understand token positions

2. Multi-Head Attention
- Self-attention in encoder and decoder
- Cross-attention in decoder (encoder-decoder attention)
- 8 parallel attention heads <br>
- Scaled dot-product attention <br>

3. Feed-Forward Networks
- ReLU (Rectified Linear Unit)
- Two linear transformations with expansion factor 4
- Position-wise fully connected layers

4. Normalization and Regularization
- Post-Layer Normalization (after residual connections)
- Dropout applied to attention and FFN outputs
- Residual Connections around each sub-layer

5. Masking Mechanisms
- Source Padding Mask: Hides padding tokens in encoder
- Target Causal Mask: Prevents attending to future tokens
- Target Padding Mask: Hides padding tokens in decoder

### Additional Info: 
##### Experimental scripts:
cuda_test.py (test if cuda is available) <br>
train_wmt14.py (train & test transformer on WMT14 dataset, currently has cuda compatability issues) 

##### Hardware Used:
CPU: 13th Gen Intel(R) Core(TM) i7-13700K (3.40 GHz) <br>
GPU: NVIDIA GeForce RTX 4070

##### References:
https://github.com/retrogtx/attention-is-all-you-need/ <br>
https://github.com/hyunwoongko/transformer/
