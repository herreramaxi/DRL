import torch

def is_cuda_available():    
    cuda_avialable = torch.cuda.is_available()
    print(f"cuda_avialable: {cuda_avialable}")

    if cuda_avialable:
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
        
    return cuda_avialable
