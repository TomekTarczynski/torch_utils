import torch
from torch import nn, optim
from torch.utils.data import DataLoader

def check_training_memory(
    model: nn.Module, 
    dataset: torch.utils.data.Dataset, 
    optimizer: optim.Optimizer, 
    criterion: nn.modules.loss._Loss, 
    batch_size: int, 
    device: torch.device, 
    max_iter: int) -> bool:
    """
    The function performs few training iteration to check whether the training with given batch_size is possible.
    Args:
        model: Model to train
        dataset: Dataset from which batches will be drawn
        optimizer: optimizer used during training
        criterion: loss function used during training
        batch_size: the size of the batch
        device: device used to store model
        max_iter: number of batches after which training will stop.

    Returns:
        bool: Indicates whether training is possible
    """
    
    try:
        model.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        loader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            drop_last=True)
    
        for i, (input, target) in enumerate(loader):
            if i>=max_iter:
                break
            input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
            logits = model(input)
            loss = criterion(logits, target)
            loss.backward()
            # optimizer.step() Optimiser step use neglibable amount of memory
            optimizer.zero_grad(set_to_none=True)

    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        torch.cuda.empty_cache()
    return True