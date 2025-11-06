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


def gradient_noise_scale(
    model: nn.Module, 
    dataset: torch.utils.data.dataset.Dataset, 
    criterion: torch.nn.modules.module.Module, 
    n: int, 
    k: int) -> float:
    """
    Function calculates gradient noise scale, which is defined as S = E[Var(teta)] / E[Mean(teta)^2].
    High values indicates that the noise in gradients introduced by random batches is relatively high in comparison to the mean value of the gradient.
    From the noise perspective there is no point in having higher batch size than the one for which S ~= 1.

    Args:
        model: Model for which gradient noise scale is calculate
        dataset: Dataset used for calculation (use training)
        criterion: The loss function
        n: Number of experiment
        k: Batch size used to estimate S. It is recommented to use at least 128. For lower values it is hard to estimate the variance of the gradient.

    Returns:
        float: The gradient noise scale
    """
    
    model.eval()
    loader=DataLoader(
        dataset=dataset,
        batch_size=k,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        drop_last=True)
    
    grads = []
    data_iter = iter(loader)
    for i in range(n):
        try:
            data, label = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            data, label = next(data_iter) 
        model.zero_grad()
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        grad_vector = torch.cat([p.grad.detach().flatten() for p in model.parameters()])
        grads.append(grad_vector)
    G = torch.stack(grads)
    grad_var = torch.var(G, dim=0)
    grad_mean = torch.mean(G, dim=0)
    S = grad_var.mean() / grad_mean.pow(2).mean()
    return S.item()                