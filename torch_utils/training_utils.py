import time
import sys, os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from cuda_utils import cuda_free_memory

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

def find_max_train_batch(
    model: nn.Module, 
    dataset: torch.utils.data.Dataset, 
    optimizer: optim.Optimizer, 
    criterion: nn.modules.loss._Loss, 
    device: torch.device,
    verbose: bool = True) -> int:
    """
    The function returns maximum batch_size which fits into the memory and allow to train the model.
    Args:
        model: Model to train
        dataset: Dataset from which batches will be drawn
        optimizer: optimizer used during training
        criterion: loss function used during training
        device: device used to store model
        verbose: If yes then print current batch_size that was evaluated.

    Returns:
        int: Max batch_size for training loop
    """
    min_batch = 1
    max_batch = len(dataset)

    model.to(device)
    while max_batch - min_batch > 1:
        current_batch = (max_batch + min_batch) // 2
    
        current_batch_result = check_training_memory(
            model = model,
            dataset = dataset,
            optimizer = optimizer,
            criterion = criterion,
            batch_size = current_batch,
            device = device,
            max_iter = 5)
    
        print(f"Current batch = {current_batch} Result: {current_batch_result}")
    
        if current_batch_result:
            min_batch = current_batch
        else:
            max_batch = current_batch
    
        cuda_free_memory()
    return min_batch   

def calculate_throughput(
    model: nn.Module, 
    dataset: torch.utils.data.Dataset, 
    optimizer: optim.Optimizer, 
    criterion: nn.modules.loss._Loss, 
    batch_size: int, 
    device: torch.device, 
    max_time_s: float) -> float:
    """
    The function calculates throughput during training for a given batch_size.
    Args:
        model: Model to train.
        dataset: Dataset from which batches will be drawn.
        optimizer: optimizer used during training.
        criterion: loss function used during training.
        batch_size: the size of the batch.
        device: device used to store model.
        max_time_s: maximum time in seconds.

    Returns:
        float: Throughput during training
    """
    
    model.train()
    optimizer.zero_grad()
    loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        drop_last = True)

    total_obs = 0
    start_time = time.time()
    while True:
        for i, (input, target) in enumerate(loader):
            input, target = input.to(device), target.to(device)
            logits = model(input)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_obs += len(input)
            if time.time() - start_time > max_time_s:
                throughput = total_obs / (time.time() - start_time)
                return throughput    