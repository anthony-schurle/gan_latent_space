import torch

class Device:
    def __init__(self):
        self.device = None

    def get_device(self):
        if self.device:
            return self.device
        elif torch.cuda.is_available():
            print("DEBUG: CUDA USED")
            self.device = torch.device("cuda")
        else:
            print("DEBUG: CPU USED")
            self.device = torch.device("cpu")
        return self.device