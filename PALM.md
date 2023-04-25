## PaLM: Scaling Language Modeling with Pathways

Improvements:
1. PaLM stands for Pathways Language Model, but this model is actually quite different from the Pathways model that Jeff Dean promoted last year. It is not a multitask/multimodal model and does not have sparse activation/dynamic routing. It is still an SPMD model with the following structure:
2. Pure decoder, similar to the structure of GPT-3, **a dense model** with 8B/62B/540B parameters.
3. Uses the SwiGLU activation function: Swish(xW) xV. **This activation function has a higher computational cost, but provides greater precision gains.**
4. Parallel Layers: y = x + MLP(LayerNorm(x + Attention(LayerNorm(x)))->y = x + MLP(LayerNorm(x)) + Attention(LayerNorm(x)). The algorithm has been changed to **use the operator fusion of MLP+Attention, which speeds up the model by 15% with a small impact on accuracy**.
5. Multi-Query Attention: single-headed key and value, multi-headed query. The algorithm has been changed to reduce the computational cost of Attention by approximately 2/3.
    * Standard multi-head attention is not efficient on accelerator hardware during autoregressive decoding because the key/value tensors are not shared between examples. In this model, the key/value mappings are shared by each head, while the queries are independent of each other. This method improves the autoregressive decoding time of the decoder.
6. RoPE Embedding: **rotation-based relative position encoding. The algorithm has been changed to improve accuracy (more friendly to long sequences).** This was also used in Llama.
7. No bias, no dropout. **This approach is increasingly being used by larger models to increase training stability.**
    * This can increase the training stability of larger models.
8. Adafactor: slightly affects accuracy and reduces optimizer states to save memory.
9. Optimized vocabulary: SentencePiece is used (a statistical method that takes frequently occurring strings as words and forms a vocabulary for segmentation), resulting in larger segments. A token table of 256K is used and text outside the vocabulary is split into UTF-8 characters. 

