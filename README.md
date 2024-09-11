# Sentiment Analysis by Fine-Tuning BERT

This repository demonstrates how to fine-tune a BERT model for sentiment analysis using PyTorch, AdamW optimizer, and a linear learning rate scheduler. The code is structured to support fine-tuning with cross-entropy loss, gradient clipping, and model checkpointing.

## Features
- **AdamW Optimizer**: Utilizes AdamW with a linear warmup schedule for learning rate adjustment.
- **Cross-Entropy Loss**: Used for classification tasks.
- **Gradient Clipping**: Ensures gradient norms are capped to prevent explosion.
- **Model Checkpointing**: Automatically saves the best model based on validation accuracy.

## Training Process

1. **Set Up Optimizer and Scheduler**:
   - Optimizer: AdamW (`lr=2e-5`)
   - Scheduler: Linear warmup with total training steps based on the data loader size.

2. **Loss Function**:
   - Loss: `CrossEntropyLoss` for classification tasks.

3. **Training Loop**:
   - For each epoch:
     - Perform a forward pass with the input data.
     - Compute the loss and predictions.
     - Backward propagate the loss and update the model using the optimizer.
     - Clip the gradients to prevent overflow.

4. **Evaluation Loop**:
   - After each epoch, evaluate the model on the validation set, tracking both accuracy and loss.
