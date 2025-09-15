# test_scheduler.py
import sys, inspect
import importlib

print("python:", sys.version.splitlines()[0])
import torch
print("torch:", torch.__version__)
try:
    from torch import optim
    print("optim module:", inspect.getfile(optim))
except Exception as e:
    print("optim import error:", e)

try:
    sig = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau.__init__)
    print("ReduceLROnPlateau.__init__ signature:", sig)
    # Try constructing with verbose
    opt = torch.optim.SGD([torch.zeros(1)], lr=0.1)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=1, verbose=True)
    print("Constructed ReduceLROnPlateau with verbose => OK")
except TypeError as e:
    print("TypeError:", e)
except Exception as e:
    print("Other error:", e)
