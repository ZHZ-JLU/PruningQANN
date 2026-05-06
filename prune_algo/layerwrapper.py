import torch
import torch.nn as nn
# from SNN.spike_layer import snnLinear
# from phase.phase_layer import phaseSnnLinear
# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device='cuda:0')
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out, extra=None):
        if extra is not None:
            inp = extra
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, (nn.Linear)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            if len(inp.shape) == 4:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
    
        norm_val = torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
        print(f"DEBUG: scaler_row shape: {self.scaler_row.shape}")
        print(f"DEBUG: scaler_row dim: {self.scaler_row.dim()}")
        print(f"DEBUG: norm_val shape: {norm_val.shape}")
        print(f"DEBUG: norm_val dim: {norm_val.dim()}")
        print(f"DEBUG: inp shape: {inp.shape}")
        if norm_val.dim() == 2:
            norm_val = norm_val.squeeze(0)
        print(f"DEBUG: norm_val shape: {norm_val.shape}")
        self.scaler_row += norm_val
        # self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples