# Transformer Architecture Interview Guide

## 1. Fundamentals and Mathematical Models (21 Questions)

### 1. Why does Transformer use multi-head attention mechanism? (Why not use a single head?)

**Answer:** Multi-head attention allows the model to attend to different types of information simultaneously:
- **Representation subspaces**: Different heads can focus on different aspects (syntax, semantics, positional relationships)
- **Parallel processing**: Multiple attention patterns can be learned in parallel
- **Robustness**: Reduces dependency on any single attention pattern
- **Increased capacity**: More parameters allow for richer representations without significantly increasing computational complexity per head

### 2. Why do Q and K in Transformer use different weight matrices? Why can't the same value be used for self-attention? (Note the difference from the first question)

**Answer:** 
- **Asymmetric relationships**: Q represents "what I'm looking for" while K represents "what I can provide"
- **Learning flexibility**: Different weight matrices allow the model to learn distinct query and key representations
- **Breaking symmetry**: Using the same matrix would make attention scores symmetric (A[i,j] = A[j,i]), limiting the model's expressiveness
- **Directional attention**: Enables learning of directional relationships between tokens

### 3. Why does Transformer choose dot-product for attention computation instead of addition? What are the computational complexity and effectiveness differences?

**Answer:**
- **Computational efficiency**: Dot-product can leverage optimized matrix multiplication (BLAS operations)
- **Parallelization**: Matrix operations are highly parallelizable on modern hardware
- **Complexity**: Dot-product is O(1) per pair, addition-based would require additional parameters
- **Effectiveness**: Dot-product naturally measures similarity in the learned embedding space
- **Hardware optimization**: GPUs are optimized for matrix multiplications

### 4. Why is attention scaled before softmax? (Why divide by sqrt(dk)) Derive and explain using formulas.

**Answer:**
The scaling factor √dk prevents the dot products from becoming too large, which would cause softmax to saturate.

**Mathematical derivation:**
- Let Q, K have dimensions d_k with elements ~N(0,1)
- Dot product q·k = Σ(q_i × k_i) has variance d_k
- Without scaling: softmax input has high variance → gradients vanish
- With scaling: variance becomes 1, maintaining gradient flow

**Formula:** Attention(Q,K,V) = softmax(QK^T/√d_k)V

### 5. How is padding handled when computing attention scores for masked operations?

**Answer:**
- **Mask creation**: Create boolean mask indicating padding positions
- **Score masking**: Set attention scores to -∞ (or very large negative value) for padding positions
- **Softmax effect**: After softmax, these positions become ~0, effectively ignoring padding
- **Implementation**: Usually done before softmax application

### 6. Why is dimension reduction needed for each head in multi-head attention?

**Answer:**
- **Computational efficiency**: Maintains overall parameter count similar to single-head attention
- **Specialization**: Each head can specialize in different aspects with smaller dimension
- **Concatenation**: Reduced dimensions allow concatenation to original model dimension
- **Mathematical**: If d_model = 512 and h = 8 heads, each head uses d_k = d_v = 64

### 7. Explain the Encoder module of Transformer.

**Answer:**
The Encoder consists of:
- **Multi-head self-attention**: Allows each position to attend to all positions
- **Add & Norm**: Residual connection + Layer normalization
- **Feed-forward network**: Two linear transformations with ReLU: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
- **Add & Norm**: Second residual connection + normalization
- **Stacking**: Typically 6 identical layers stacked

### 8. Why is distance scaled by embedding_size after obtaining input word vectors?

**Answer:**
- **Scale matching**: Ensures positional encodings and word embeddings are in similar magnitude ranges
- **Learning stability**: Prevents one component from dominating during training
- **Gradient flow**: Maintains proper gradient scaling throughout the network
- **Original paper**: Empirically found to improve performance

### 9. Brief introduction to Transformer's positional encoding.

**Answer:**
Uses sinusoidal functions to encode position information:
- **Formula**: PE(pos,2i) = sin(pos/10000^(2i/d_model)), PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
- **Properties**: 
  - Deterministic and unique for each position
  - Can handle sequences longer than training sequences
  - Linear relationships between positions can be learned
- **Alternative**: Learned positional embeddings (used in some variants)

### 10. What techniques are known about positional encoding?

**Answer:**
- **Sinusoidal encoding**: Original Transformer approach
- **Learned positional embeddings**: Trainable position vectors
- **Relative positional encoding**: Focus on relative distances rather than absolute positions
- **Rotary Position Embedding (RoPE)**: Used in models like GPT-J
- **ALiBi (Attention with Linear Biases)**: Bias attention scores based on distance

### 11. Brief explanation of residual connections in Transformer and their significance.

**Answer:**
- **Structure**: x_output = LayerNorm(x_input + Sublayer(x_input))
- **Benefits**:
  - Mitigates vanishing gradient problem
  - Enables training of deeper networks
  - Provides gradient highway for backpropagation
  - Stabilizes training dynamics
- **Placement**: Applied around both attention and feed-forward sublayers

### 12. Why does Transformer use LayerNorm instead of BatchNorm?

**Answer:**
- **Sequence independence**: LayerNorm doesn't depend on batch statistics, better for variable-length sequences
- **Training stability**: More stable for sequence models with varying lengths
- **Inference consistency**: No dependency on batch composition during inference
- **Per-sample normalization**: Normalizes across feature dimension for each sample independently

## 2. Training and Optimization (19 Questions)

### 1. Where do the trainable parameters for Queries, Keys, and Values come from in Transformer?

**Answer:**
- **Linear projection matrices**: W_Q, W_K, W_V are learned weight matrices
- **Dimensions**: Input dimension d_model → head dimension d_k (or d_v)
- **Per head**: Each head has its own set of projection matrices
- **Training**: Updated through backpropagation during training
- **Initialization**: Typically initialized with small random values

### 2. What exactly is being trained in Transformer's FeedForward layer?

**Answer:**
The FFN trains two linear transformations:
- **First layer**: d_model → d_ff (typically 4× larger)
- **Second layer**: d_ff → d_model
- **Parameters**: Weight matrices W₁, W₂ and bias terms b₁, b₂
- **Purpose**: Processes each position independently, adds non-linearity
- **Activation**: Usually ReLU or GELU between the layers

### 3. Analyze the computational complexity of Transformer's Embeddings, Attention, and Feedforward layers.

**Answer:**
- **Embeddings**: O(n × d_model) - linear in sequence length
- **Self-Attention**: O(n² × d_model) - quadratic in sequence length
- **Feed-Forward**: O(n × d_model × d_ff) - linear in sequence length
- **Bottleneck**: Self-attention becomes the computational bottleneck for long sequences
- **Memory**: Attention requires O(n²) memory for storing attention weights

### 4. How does Transformer's Positional Encoding express relative positional relationships?

**Answer:**
- **Sinusoidal properties**: Different frequencies allow the model to learn relative distances
- **Linear combinations**: The model can learn linear transformations to identify relative positions
- **Distance encoding**: PE(pos+k) can be expressed as a linear function of PE(pos)
- **Learning**: The model learns to utilize these patterns during training

### 5. What assumption does LayerNormalization make about neural networks?

**Answer:**
LayerNorm assumes:
- **Feature independence**: Features within a layer should have similar importance
- **Mean-zero, unit-variance**: Optimal activation distribution for gradient flow
- **Per-sample normalization**: Each sample should be normalized independently
- **Learned scale and shift**: The model should learn appropriate scale (γ) and shift (β) parameters

### 6. Analyze the dependency relationship between Decoder and Encoder in Transformer from a data perspective.

**Answer:**
- **Cross-attention dependency**: Decoder queries attend to encoder outputs (keys and values)
- **Information flow**: Encoder provides context, decoder generates sequence
- **Asymmetric**: Encoder processes all positions simultaneously, decoder is autoregressive
- **Training vs. Inference**: 
  - Training: Teacher forcing allows parallel processing
  - Inference: Sequential generation creates dependency chain

### 7. Describe Transformer's Tokenization mathematical principles, workflow, problems, and improvements.

**Answer:**
- **Subword tokenization**: Breaks words into smaller units (BPE, WordPiece, SentencePiece)
- **Workflow**: 
  1. Pre-tokenization (split by whitespace/punctuation)
  2. Apply subword algorithm
  3. Convert to token IDs
- **Problems**: 
  - Out-of-vocabulary handling
  - Language-specific biases
  - Inconsistent tokenization across languages
- **Improvements**: 
  - Byte-level BPE
  - Language-agnostic tokenizers
  - Dynamic tokenization

### 8. Describe methods to reduce self-attention complexity from O(n²) to O(n).

**Answer:**
- **Sparse attention**: Only attend to subset of positions (local windows, strided patterns)
- **Linear attention**: Use kernel methods to approximate attention
- **Low-rank approximation**: Factorize attention matrix
- **Linformer**: Project sequence length dimension
- **Performer**: Use random feature maps
- **Longformer**: Combination of local and global attention

### 9. Can BERT's CLS token effectively represent sentence embeddings?

**Answer:**
- **Design purpose**: CLS token aggregates information from entire sequence through self-attention
- **Effectiveness**: Works well for many tasks but may not capture all nuances
- **Limitations**: 
  - Single vector may lose information
  - Task-dependent effectiveness
- **Alternatives**: 
  - Mean pooling of all tokens
  - Weighted pooling
  - Task-specific pooling strategies

## 3. Applications and Practice (6 Questions)

### 1. How to implement Zero-shot Learning using Transformer?

**Answer:**
- **Pre-training**: Train on large diverse dataset to learn general representations
- **Prompt engineering**: Frame new tasks as text completion or classification
- **In-context learning**: Provide examples in the input context
- **Task formatting**: Convert target task into natural language format
- **Examples**: GPT-3 style prompting, T5's text-to-text format

### 2. Describe at least 2 methods for comparing similarity between embeddings from different trained models.

**Answer:**
- **Cosine similarity**: Measure angular distance between embedding vectors
- **Centered Kernel Alignment (CKA)**: Compare representation similarities across layers
- **Procrustes analysis**: Align embedding spaces through orthogonal transformation
- **Canonical Correlation Analysis**: Find linear relationships between embedding spaces

### 3. How to make small models (like LSTM) have capabilities of large models (like BERT)?

**Answer:**
- **Knowledge distillation**: Train small model to mimic large model's outputs
- **Transfer learning**: Fine-tune pre-trained representations
- **Architecture improvements**: Use attention mechanisms in smaller models
- **Data augmentation**: Increase training data diversity
- **Ensemble methods**: Combine multiple small models
- **Compression techniques**: Pruning, quantization of large models

## 4. Technical Deep Dive and Innovation (29 Questions)

### 1. Mathematically explain methods for masking arbitrary positions and lengths in Transformer.

**Answer:**
- **Attention masking**: Set attention scores to -∞ for masked positions
- **Causal masking**: Upper triangular mask for autoregressive generation
- **Padding masking**: Mask padded positions in variable-length sequences
- **Mathematical**: M[i,j] = 0 if position j should be attended to by position i, -∞ otherwise
- **Implementation**: Apply mask before softmax: softmax(QK^T/√d_k + M)

### 2. Describe differences in attention mechanisms between Encoder and Decoder.

**Answer:**
- **Encoder**: Bidirectional self-attention (can attend to all positions)
- **Decoder**: 
  - Masked self-attention (causal/autoregressive)
  - Cross-attention to encoder outputs
- **Masking**: Decoder uses causal mask to prevent attending to future positions
- **Information flow**: Encoder processes in parallel, decoder is sequential during inference

### 3. Describe Transformer Decoder's embedding layer architecture design, workflow, and mathematical principles.

**Answer:**
- **Token embeddings**: Convert token IDs to dense vectors
- **Positional encodings**: Add position information
- **Embedding scaling**: Multiply by √d_model for magnitude matching
- **Workflow**: Token ID → Embedding lookup → Add positional encoding → Layer input
- **Mathematics**: E = √d_model × Embed(token) + PE(position)

### 4. Describe how embedding is performed in the Decoder during Transformer training lifecycle.

**Answer:**
- **Teacher forcing**: Use ground truth tokens during training
- **Shifted input**: Decoder input is target sequence shifted by one position
- **Embedding lookup**: Convert token IDs to embeddings at each position
- **Parallel processing**: All positions processed simultaneously during training
- **Loss computation**: Compare predictions with actual next tokens

### 5. Describe how embedding is performed in the Decoder during Transformer inference lifecycle.

**Answer:**
- **Autoregressive generation**: Generate one token at a time
- **Previous tokens**: Use previously generated tokens as input
- **Incremental**: Each step adds one more token to the sequence
- **Caching**: Key-value caching optimizes repeated computations
- **Stopping**: Continue until special end token or maximum length

### 6. What are the drawbacks if Transformer uses the same process for Training and Inference?

**Answer:**
- **Exposure bias**: Model never sees its own mistakes during training
- **Training-inference mismatch**: Different input distributions
- **Error accumulation**: Mistakes compound during sequential generation
- **Slower training**: Sequential processing would be much slower
- **Solutions**: Scheduled sampling, curriculum learning

### 7. Why are Transformer's matrix dimensions 3D?

**Answer:**
The three dimensions represent:
- **Batch dimension**: Number of sequences processed in parallel
- **Sequence dimension**: Length of each sequence (number of tokens)
- **Feature dimension**: Embedding/hidden state size
- **Format**: [batch_size, sequence_length, hidden_size]
- **Efficiency**: Enables batched processing for parallel computation

### 8. Describe where attention is used in Encoder-Decoder Transformer and its functions.

**Answer:**
- **Encoder self-attention**: Each position attends to all positions in input
- **Decoder self-attention**: Masked attention to previous positions
- **Cross-attention**: Decoder attends to encoder outputs
- **Functions**:
  - Information aggregation
  - Long-range dependencies
  - Context integration
  - Sequence-to-sequence alignment

### 9. Describe the function and mathematical implementation of masking in Transformer attention mechanism during training and inference.

**Answer:**
- **Training masks**:
  - Padding mask: Ignore padded positions
  - Causal mask: Prevent future information leakage
- **Inference masks**: Same causal masking for generation
- **Implementation**: 
  ```
  masked_scores = scores + mask
  attention_weights = softmax(masked_scores)
  ```
- **Mask values**: -∞ for masked positions, 0 for valid positions

### 10. Describe Transformer's training loss workflow and mathematical formulas.

**Answer:**
- **Cross-entropy loss**: L = -Σ log P(y_i | x, y_<i)
- **Teacher forcing**: Use ground truth for all positions
- **Parallel computation**: All positions computed simultaneously
- **Gradient flow**: Backpropagate through all layers
- **Optimization**: Typically Adam optimizer with learning rate scheduling

### 11. Explain QKV partition in multi-head attention through linear layer computation.

**Answer:**
- **Linear projections**: Q = XW_Q, K = XW_K, V = XW_V
- **Dimension**: W_Q, W_K, W_V are [d_model, d_model] matrices
- **Head splitting**: Reshape to [batch, seq_len, num_heads, head_dim]
- **Parallel processing**: Each head processes independently
- **Concatenation**: Outputs concatenated and projected back to d_model

## 5. Performance Optimization and Model Improvement (7 Questions)

### 1. Issues with using inference process for Transformer training?

**Answer:**
- **Sequential bottleneck**: Training would be extremely slow
- **No parallelization**: Cannot leverage parallel processing capabilities
- **Exposure bias**: Model trained on perfect inputs, tested on imperfect ones
- **Memory inefficiency**: Cannot batch process effectively
- **Gradient flow**: More complex gradient computation paths

### 2. Why are Transformer's matrix dimensions 3D?

**Answer:**
Already covered in Technical Deep Dive section #7.

### 3. Describe where attention is used in Encoder and Decoder Transformer and their functions.

**Answer:**
Already covered in Technical Deep Dive section #8.

### 4. Function and mathematical implementation of masking in Transformer attention mechanism during training and inference.

**Answer:**
Already covered in Technical Deep Dive section #9.

### 5. Describe Transformer's training loss workflow and mathematical formulas.

**Answer:**
Already covered in Technical Deep Dive section #10.

### 6. Explain QKV partition computation through linear layers in multi-head attention.

**Answer:**
Already covered in Technical Deep Dive section #11.

## 6. Specific Application Scenarios and Solutions (2 Questions)

### 1. How to use Transformer for dialogue system to determine if user communication content is off-topic?

**Answer:**
- **Classification approach**: Fine-tune Transformer for binary classification (on-topic/off-topic)
- **Context encoding**: Encode conversation history and current utterance
- **Similarity measurement**: Compare current input with topic embeddings
- **Threshold-based**: Set confidence threshold for off-topic detection
- **Training data**: Collect labeled examples of on-topic and off-topic conversations

### 2. Settings for learning rate and dropout during Transformer training?

**Answer:**
- **Learning rate**: 
  - Warmup schedule: Start low, increase to peak, then decay
  - Typical peak: 1e-4 to 1e-3
  - Warmup steps: 4000-10000 steps
- **Dropout**: 
  - Attention dropout: 0.1
  - Residual dropout: 0.1
  - Embedding dropout: 0.1
- **Adaptive**: Adjust based on model size and dataset

## Large Model Fundamentals

### 1. What are the current mainstream open-source model architectures?

**Answer:**
- **GPT family**: GPT-2, GPT-3, GPT-J, GPT-NeoX
- **BERT family**: BERT, RoBERTa, DeBERTa, ELECTRA
- **T5**: Text-to-Text Transfer Transformer
- **LLaMA**: Meta's large language model
- **PaLM**: Pathways Language Model architecture
- **Switch Transformer**: Sparse expert models

### 2. What are the differences between prefix decoder, causal decoder, and encoder-decoder?

**Answer:**
- **Prefix decoder**: Can attend to prefix bidirectionally, then causally (PaLM)
- **Causal decoder**: Pure autoregressive, only attends to previous tokens (GPT)
- **Encoder-decoder**: Separate encoding and decoding phases (T5, BART)
- **Use cases**: 
  - Causal: Text generation
  - Encoder-decoder: Translation, summarization
  - Prefix: Hybrid tasks

### 3. What is the training objective for large language models?

**Answer:**
- **Next token prediction**: Predict the next token given previous context
- **Masked language modeling**: Predict masked tokens (BERT-style)
- **Text-to-text**: Convert all tasks to text generation format
- **Mathematical**: Maximize P(x_t | x_<t) for autoregressive models

### 4. What causes hallucination?

**Answer:**
- **Training data**: Inconsistent or incorrect information in training data
- **Overconfident generation**: Model generates plausible but incorrect information
- **Context limitations**: Insufficient context for accurate responses
- **Optimization objective**: Training for fluency vs. factual accuracy
- **Knowledge boundaries**: Model extrapolates beyond training knowledge

### 5. Why are most current large models decoder-only architectures?

**Answer:**
- **Simplicity**: Single architecture for all tasks
- **Scalability**: Easier to scale to very large sizes
- **Versatility**: Can handle both understanding and generation
- **Training efficiency**: Single training objective (next token prediction)
- **In-context learning**: Naturally supports few-shot learning
- **Success**: Proven effectiveness of GPT-style models