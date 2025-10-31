import torch
import torch.nn as nn
import torch.nn.functional as F

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantizer(nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()

    def forward(self, input):
        clipped_input = torch.clamp(input, min=-1.49, max=1.49)
        return RoundSTE.apply(clipped_input)

class LatentQuantizer:
    def __init__(self, bit: int):
        super(LatentQuantizer, self).__init__()
        self.bit = bit
        self.levels = 2**bit
        self.min_val = None
        self.max_val = None
        self.scale = None

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.min_val = tensor.min().item()
        self.max_val = tensor.max().item()
        if self.max_val == self.min_val:
            return tensor
        else:
            self.scale =(self.levels-1) / (self.max_val - self.min_val)
            quantizedTensor = (tensor - self.min_val) * self.scale + 0.5
            OutTensor = FloorSTE.apply(quantizedTensor)
            return OutTensor

    def dequantized(self, quantizedTensor:torch.Tensor) -> torch.Tensor:
        if self.min_val is None or self.max_val is None or self.scale is None:
            return quantizedTensor
        else:
            reTensor = quantizedTensor / self.scale + self.min_val
            return reTensor

    def train(self, input_vector, qp):
        self.bit = qp
        self.levels = 2**qp
        q = self.forward(input_vector)
        d = self.dequantized(q)
        return q, d

    def reconstruct_infer(self, q):
        if self.min_val is None or self.max_val is None or self.scale is None:
            raise ValueError("Parameters max, min and scale must be trained or loaded before inference.")
        r = q/self.scale + self.min_val
        return r

    def save(self, filepath):
        if self.min_val is None or self.max_val is None or self.scale is None:
            raise ValueError("Parameters min, max and scale are not initialized!")
        model_params = {
            "a": self.min_val,
            "b": self.max_val,
            "scale": self.scale
        }
        torch.save(model_params, filepath)

    def load(self, filepath):
            checkpoint = torch.load(filepath)
            self.min_val = checkpoint["a"]
            self.max_val = checkpoint["b"]
            self.scale = checkpoint["scale"]
            print(f"Model parameters loaded from {filepath}")

    def toCUDA(self, device):
            self.min_val = self.min_val.to(device)
            self.max_val = self.max_val.to(device)
            self.scale = self.scale.to(device)
class RobustQuantize:
    def __init__(self, lmd = 1e-2):
        self.lmd = lmd
        self.a = None
        self.b = None
    def quantize(self, x, bits, axis=0, eps = 1e-8):
        max_value, _ = torch.max(x, dim = axis, keepdim = True)
        min_value, _ = torch.min(x, dim = axis, keepdim = True)
        scaled_x = (x - min_value)/(max_value - min_value+ eps)*(2**bits - 1)
        # delta_x = (scaled_x.round() - scaled_x).detach()
        delta_x = RoundSTE.apply(scaled_x) - scaled_x
        q = scaled_x + delta_x
        return q
    def reconstruct_train(self, q, x, axis = 0):
        E_q2 = torch.mean(q**2, dim = axis, keepdim=True)
        E_q = torch.mean(q, dim=axis, keepdim=True)
        E_qx = torch.mean(q*x, dim=axis, keepdim=True)
        E_x = torch.mean(x, dim=axis, keepdim=True)

        Var_q = E_q2 - E_q**2
        Cov_qx = E_qx - E_q * E_x

        self.a = Cov_qx / (Var_q + self.lmd)
        self.b = E_x - self.a * E_q

        r = self.a * q + self.b

        return r
    def reconstruct_infer(self, q):
        if self.a is None or self.b is None:
            raise ValueError("Parameters a and b must be trained or loaded before inference.")
        r = self.a * q + self.b
        return r

    def train(self, x, bits, axis=0, eps=1e-8):
        q = self.quantize(x, bits, axis, eps)
        r = self.reconstruct_train(q, x, axis)
        return q, r

    def infer(self, q):
        return self.reconstruct_infer(q)

    def save(self, filepath):
        if self.a is None or self.b is None:
            raise ValueError("Parameters a and b are not initialized!")
        model_params = {
            "a": self.a.cpu() if self.a.is_cuda else self.a,
            "b": self.b.cpu() if self.b.is_cuda else self.b,
            "lmd": self.lmd
        }
        torch.save(model_params, filepath)
        # print(f"Model parameters saved to {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.a = checkpoint["a"]
        self.b = checkpoint["b"]
        self.lmd = checkpoint.get("lmd", 1e-2)
        print(f"Model parameters loaded from {filepath}")

    def toCUDA(self, device):
        self.a = self.a.to(device)
        self.b = self.b.to(device)





if __name__ == "__main__":
    a = nn.Parameter(torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.6, 1.0]]))
    b = 2
    quantizer = LatentQuantizer(b)
    print(a)
    quantized_a = quantizer(a)
    print(quantized_a)

    de_q = quantizer.dequantized(quantized_a)
    print(de_q)

    quantize_model = RobustQuantize()
    latent_int, latent_rec =   quantize_model.train(a, b)
    print(latent_int)
    print(latent_rec)