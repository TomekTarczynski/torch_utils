import torch
import yaml

import subprocess
import logging
from importlib import resources


TERA = 1e12

def get_power_limit() -> float:
    """
    Retrieve the max GPU Power Limit [W] using nvidia-smi

    Returns:
        float: Power limit in watts.
    """
    
    result = subprocess.run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits", "--id=0"], capture_output=True, text=True, check=True)
    logging.debug(f"Result in get_power_limit():\n{result.stdout}")
    return float(result.stdout.strip())

def get_sm_clock() -> float:
    """
    Retrieve max clock frequency of a GPU core.

    Returns:
        float: Max clock frequency in MHz
    """
    
    result = subprocess.run(["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits", "--id=0"], capture_output=True, text=True, check=True)
    logging.debug(f"Result in get_sm_clock():\n{result.stdout}")
    return float(result.stdout.strip())

def get_device_information(sm_cores: dict) -> dict:
    """
    Retrieves all essential information about the GPU.

    Args:
        sm_cores (dict): The dictionary that maps compute capabilities into number of cores per Streaming Multiprocessor

    Returns:
        dict: Details about the GPU
    """
    
    props = torch.cuda.get_device_properties(0)
    
    device_information = {
        "device_name": torch.cuda.get_device_name(0),
        "number_of_sm": props.multi_processor_count,
        "power_limit": {
            "value": get_power_limit(),
            "unit": "W"
        },
        "compute_capability": (props.major, props.minor),
        "cores_per_sm": sm_cores['cuda_cores_per_sm'][f"{props.major}.{props.minor}"],
        "sm_clock": {
            "value": get_sm_clock(),
            "unit": "MHz"
        },
        "memory": {
            "value": round(torch.cuda.mem_get_info()[1] / (1024**3), 2),
            "unit": "GiB"
        }
    }
    device_information["total_number_of_cores"] = device_information["number_of_sm"] * device_information["cores_per_sm"]
    device_information["theoretical_flops"] = {
        "value": round((device_information["total_number_of_cores"] * device_information["sm_clock"]["value"] * 2 * 1e6) / TERA, 2), # Multpied by 2 because add and multiply are made in a single operation
        "unit": "tflops",
        "precision": "float32"
    }
    
    logging.info(f"Device information:\n{device_information}")    
    return device_information

def load_config() -> dict:
    """
    Loads a dictionary containing information about number of cores in Streamline Multiprocessors.

    Returns:
        dict: Number of cores in SM.
    """

    with resources.files("torch_utils").joinpath("data/sm_cores.yaml").open("r") as file:
        sm_cores = yaml.safe_load(file)
        logging.info("sm_cores.yaml loaded successfully")
        return sm_cores