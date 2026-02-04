# Shakespeare-GPT: A Minimalist Word2vec Implementation

A lightweight, from-scratch implementation of a Word2vec-based language model trained on the corpus of Shakespeare's *Romeo and Juliet*. This project demonstrates the transition from basic neural concepts to optimized matrix-based computation.

## Key Features
- **Pure NumPy Implementation:** No high-level ML frameworks (PyTorch/TensorFlow) used.
- **Vectorized Training:** Highly optimized matrix operations replacing nested loops for 10x+ speedup.
- **Advanced Generation:** Implements **Temperature Scaling** and **Repetition Penalty** to control creativity and avoid infinite loops.
- **Semantic Vector Space:** Captures complex relationships (e.g., `i` + `have` results in `francis`, reflecting dialogue patterns).

## Technical Specifications
- **Architecture:** Continuous Bag of Words (CBOW) inspired linear model.
- **Embedding Dimensions ($k$):** 128
- **Context Window:** 5 (Total 10 surrounding words)
- **Optimizer:** Mini-batch Gradient Descent (Batch size: 512)
- **Final Loss:** ~5.9 after 500 epochs.

## How to Use
1. **Training:**
   ```bash
   python train.py