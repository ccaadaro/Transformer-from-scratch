# Transformer - Use Cases and Testing

This project implements a Transformer model from scratch with multiple use cases to demonstrate its capabilities.

## Project Files

- `transformer.py` - Main Transformer model implementation
- `test_cases.py` - Basic and specific use cases
- `advanced_cases.py` - Advanced use cases
- `README.md` - This documentation

## Model Structure

The model implements the complete Transformer architecture with:

### Main Components
- **Positional Encoding**: Positional encoding for sequences
- **Multi-Head Attention**: Multi-head attention mechanism
- **Feed Forward Networks**: Feed forward neural networks
- **Encoder and Decoder Layers**: Encoding and decoding layers
- **Layer Normalization**: Layer normalization
- **Dropout**: Regularization

### Technical Features
- GPU support (CUDA)
- Attention masks for padding and causality
- Greedy decoding for inference
- Adam optimizer with adaptive learning rate

## Implemented Use Cases

### 1. Basic Cases (`test_cases.py`)

#### Sequence Copying
- **Objective**: Copy input sequence exactly
- **Input**: `[2, 5, 8, 3, 12, 7, 4, 9]`
- **Output**: `[2, 5, 8, 3, 12, 7, 4, 9]`
- **Accuracy**: 100%

#### Sequence Reversal
- **Objective**: Reverse the order of the sequence
- **Input**: `[2, 5, 8, 3, 12, 7]`
- **Output**: `[7, 12, 3, 8, 5, 2]`
- **Accuracy**: 100%

#### Arithmetic Sequences
- **Objective**: Continue arithmetic patterns
- **Example**: `[2, 4, 6] → [8, 10, 12]`
- **Accuracy**: 100%

#### Memory Test
- **Objective**: Remember first element of long sequences
- **Range**: Sequences of 5 to 20 elements
- **Accuracy**: 100% up to 20 elements

### 2. Advanced Cases (`advanced_cases.py`)

#### Pattern Recognition
- **Fibonacci**: `[1, 1, 2] → [3, 5, 8]`
- **Squares**: `[4, 9, 16] → [25, 36, 49]`
- **Prime Numbers**: `[3, 5, 7] → [11, 13, 17]`
- **Powers of 2**: `[2, 4, 8] → [16, 32, 64]`

#### Sorting
- **Objective**: Sort numeric sequences
- **Example**: `[5, 2, 8, 1, 6] → [1, 2, 5, 6, 8]`

#### Mathematical Operations
- **Addition**: `[15, +, 23] → [38]`
- **Format**: `[a, 100, b] → [result]` (100 represents '+')

## How to Run

### Requirements
```bash
pip install torch
```

### Running Tests

#### Base Model
```bash
python transformer.py
```

#### Specific Cases
```bash
python test_cases.py
```

#### Advanced Cases
```bash
python advanced_cases.py
```

## Performance Results

### Basic Cases
- **Sequence Copying**: 100% accuracy
- **Reversal**: 100% accuracy
- **Arithmetic Sequences**: 100% accuracy
- **Long-term Memory**: 100% up to 20 elements

### Advanced Cases
- **Mathematical Patterns**: Variable according to complexity
- **Sorting**: High accuracy on short sequences
- **Mathematical Operations**: High accuracy on simple additions

## Model Configuration

### Typical Parameters
```python
model = Transformer(
    src_vocab=1000,        # Input vocabulary size
    tgt_vocab=1000,        # Output vocabulary size
    d_model=256,           # Model dimension
    num_heads=8,           # Number of attention heads
    num_layers=4,          # Number of layers
    d_ff=512,              # Feedforward dimension
    dropout=0.1,           # Dropout rate
    max_len=100            # Maximum sequence length
)
```

### Training Hyperparameters
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)
batch_size = 32
```

## Technical Features

### Mask Handling
- **Padding Mask**: Ignores padding tokens (0)
- **Causal Mask**: Prevents decoder from seeing future tokens
- **Source Mask**: Mask for input sequence

### Special Tokens
- `0`: Padding/End token
- `1`: Start token
- `2-vocab_size`: Content tokens

### Decoding
- **Greedy Decoding**: Selects most probable token
- **Automatic Stopping**: Stops when generating end token
- **Maximum Length**: Limits output length

## Potential Applications

1. **Machine Translation**: English-Spanish
2. **Sequence Analysis**: Temporal patterns
3. **Text Generation**: Sequence continuation
4. **Reasoning Tasks**: Logical operations
5. **Data Processing**: Numeric transformations

## Current Limitations

- Limited vocabulary to integers
- Relatively short sequences
- Training on synthetic data
- Not optimized for production

## Future Improvements

- [ ] Real text support
- [ ] Beam search for decoding
- [ ] More sophisticated evaluation metrics
- [ ] Performance optimizations
- [ ] Attention visualization
- [ ] Checkpointing and model saving

## Contributions

This project is educational and designed to demonstrate fundamental Transformer concepts. Improvements and extensions are welcome.

## References

- Attention Is All You Need (Vaswani et al., 2017)
- The Annotated Transformer
- PyTorch Documentation

## Implemented Use Cases

### 1. English-Spanish Translation
- **File**: `transformer.py` (function `test_translation_model`)
- **Description**: Translates simple phrases from English to Spanish
- **Vocabulary**: 26 basic words in both languages
- **Examples**:
  - "hello world" → "hola mundo"
  - "good morning" → "buenos días"
  - "how are you" → "cómo estás"

### 2. Sequence Copying
- **File**: `test_cases.py` (function `test_sequence_copying`)
- **Description**: Tests the model's ability to copy sequences exactly
- **Utility**: Verify that the model can learn basic patterns

### 3. Arithmetic Sequences
- **File**: `test_cases.py` (function `test_arithmetic_sequences`)
- **Description**: Predicts next numbers in arithmetic sequences
- **Examples**:
  - [2, 4, 6] → [8, 10, 12]
  - [5, 10, 15] → [20, 25, 30]

### 4. Memory Tests
- **File**: `test_cases.py` (function `test_memory_capacity`)
- **Description**: Evaluates the model's ability to remember long-term information
- **Methodology**: Predict the first element of a sequence after seeing the entire sequence

### 5. Performance Analysis
- **File**: `evaluate_model.py`
- **Description**: Compares different model configurations
- **Metrics**:
  - Number of parameters
  - Inference time
  - Memory usage
  - Translation accuracy

## How to Run

### Basic Tests
```bash
python transformer.py
```

### Specific Use Cases
```bash
python test_cases.py
```

### Performance Evaluation
```bash
python evaluate_model.py
```

## Model Configurations

The model supports different configurations:

### Small
- `d_model`: 64
- `num_heads`: 2
- `num_layers`: 1
- `d_ff`: 128

### Medium
- `d_model`: 128
- `num_heads`: 4
- `num_layers`: 2
- `d_ff`: 256

### Large
- `d_model`: 256
- `num_heads`: 8
- `num_layers`: 3
- `d_ff`: 512

## Model Architecture

The implemented Transformer includes:

1. **Positional Embeddings**: Encode the position of each token
2. **Multi-Head Attention**: Allows the model to attend to different parts of the sequence
3. **Feed-Forward Networks**: Fully connected layers with ReLU
4. **Layer Normalization**: Normalization to stabilize training
5. **Dropout**: Regularization to prevent overfitting

## Technical Features

- **GPU Compatibility**: Automatically detects and uses CUDA if available
- **Mask Handling**: Supports padding and causal masks
- **Greedy Decoding**: Implements simple decoding for inference
- **Flexible Vocabulary**: Can work with different vocabulary sizes

## Additional Use Cases

### Attention Visualization
```python
test_attention_visualization()
```
Shows the model's attention patterns to understand what parts of the input are being processed.

### Capacity Analysis
```python
test_model_capacity()
```
Compares performance of different model configurations.

### Quality Benchmark
```python
benchmark_translation_quality()
```
Evaluates translation quality with specific metrics.

## Performance Report

The system generates a complete report in JSON format with:
- System information
- Performance metrics by configuration
- Translation quality results
- Evaluation timestamp

## Requirements

- Python 3.7+
- PyTorch 1.0+
- CUDA (optional, for GPU acceleration)

## Future Extensions

1. **Beam Search**: Implement more sophisticated decoding
2. **Attention Visualization**: Create graphs of attention patterns
3. **More Languages**: Expand vocabulary for more languages
4. **BLEU Metrics**: Implement standard translation evaluation
5. **Fine-tuning**: Allow fine-tuning on specific datasets

## Technical Notes

- The model uses triangular masks to prevent the decoder from seeing future tokens
- The loss function ignores padding tokens (index 0)
- Embeddings are scaled by √d_model as in the original paper
- Dropout is included in attention and feed-forward layers
