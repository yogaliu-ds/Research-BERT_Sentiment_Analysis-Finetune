# PyTorch Model Training with AdamW Optimizer

This repository contains a PyTorch implementation for training and evaluating a model using the AdamW optimizer and a linear learning rate scheduler. The code supports fine-tuning with cross-entropy loss and gradient clipping.

## Features
- **AdamW Optimizer**: Utilizes the AdamW optimizer with a linear warmup schedule.
- **Cross-Entropy Loss**: Designed for classification tasks.
- **Gradient Clipping**: Prevents gradient explosion by clipping gradients to a max norm.
- **Model Checkpointing**: Saves the best model based on validation accuracy.


# Training Process

    Set Up Optimizer and Scheduler:
        Optimizer: AdamW (lr=2e-5)
        Scheduler: Linear warmup with total training steps based on data loader size.

    Loss Function:loss_fn = nn.CrossEntropyLoss().to(device)

    Training Loop: Each epoch performs the following steps:
        Forward pass with input data
        Compute loss and predictions
        Backward pass and optimizer step
        Gradient clipping

    Evaluation Loop: After each epoch, the model is evaluated on the validation set to track accuracy and loss.
