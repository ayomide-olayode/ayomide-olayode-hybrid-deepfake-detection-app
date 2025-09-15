# src/overfit_single_batch.py
import torch
import numpy as np
from models.hybrid_model import create_model, get_model_config
from datasets.deepfake_dataset import create_data_loaders, get_dataset_config
from config import config
import torch.nn as nn
import time

cfg = get_dataset_config()
cfg['batch_size'] = 4
cfg['num_workers'] = 0

train_loader, _ = create_data_loaders(data_dir=str(config.DATA_DIR / "processed"), **cfg)
frames, text_data, labels, meta = next(iter(train_loader))
labels = labels.float().view(-1)

device = config.DEVICE
model = create_model(get_model_config()).to(device)
model.train()

# optionally freeze everything except classifier to check head capacity:
# for name, p in model.named_parameters():
#     if 'classifier' not in name:
#         p.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

frames = frames.to(device)
text_data = {k: v.to(device) for k, v in text_data.items()}
labels = labels.to(device)

print("Starting overfit test on single batch. If model can overfit, loss should approach 0 and probs -> {0,1}")
for step in range(200):
    optimizer.zero_grad()
    outputs = model(frames, text_data).view(-1)
    # detect probs vs logits
    mn = float(outputs.min().item())
    mx = float(outputs.max().item())
    if 0.0 <= mn and mx <= 1.0:
        # outputs are probs -> convert to logits
        eps = 1e-7
        out_clamped = torch.clamp(outputs, eps, 1-eps)
        logits = torch.log(out_clamped/(1-out_clamped))
    else:
        logits = outputs
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(logits)
    if step % 10 == 0 or step == 199:
        print(f"step {step:03d} loss {loss.item():.6f} mean_prob {probs.mean().item():.4f} std {probs.std().item():.6f}")

print("Final probs:", probs.detach().cpu().numpy())
print("Final labels:", labels.detach().cpu().numpy())
