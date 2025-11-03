import torch
import gc

def cuda_free_memory() -> None:
    """
    Performs garbage collection and realase all unused memory.
    """
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()           

def cuda_print_memory() -> None:
    """
    Prints how much VRAM is available
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved  = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    max_res   = torch.cuda.max_memory_reserved() / 1024**2

    print(f"Currently allocated: {allocated:.2f} MB")
    print(f"Currently reserved : {reserved:.2f} MB")
    print(f"Max allocated      : {max_alloc:.2f} MB")
    print(f"Max reserved       : {max_res:.2f} MB")    

def cuda_list_objects() -> None:
    """
    List all live Python objects that hold CUDA memory.
    """
    cuda_objects = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                cuda_objects.append(obj)
            elif hasattr(obj, "data") and torch.is_tensor(obj.data) and obj.data.is_cuda:
                cuda_objects.append(obj)
        except Exception:
            pass  # ignore uninspectable objects
    return cuda_objects