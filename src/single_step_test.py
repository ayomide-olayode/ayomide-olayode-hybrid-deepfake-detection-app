# single_step_test.py
import torch, numpy as np
from models.hybrid_model import create_model, get_model_config
from datasets.deepfake_dataset import create_data_loaders, get_dataset_config
from config import config

cfg = get_dataset_config()
cfg['batch_size'] = 2
cfg['num_workers'] = 0
train_loader, _ = create_data_loaders(data_dir=str(config.DATA_DIR / "processed"), **cfg)
frames, text_data, labels, meta = next(iter(train_loader))

device = torch.device('cpu')
model = create_model(get_model_config()).to(device)
model.train()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
criterion = torch.nn.BCEWithLogitsLoss()

frames = frames.to(device)
text_data = {k: v.to(device) for k, v in text_data.items()}
labels = labels.float().view(-1).to(device)

# forward
logits = model(frames, text_data).view(-1)
loss = criterion(logits, labels)
print("Initial loss:", float(loss.item()))
# backward
opt.zero_grad()
loss.backward()

# grad norms
total_norm = 0.0
for name, p in model.named_parameters():
    if p.grad is None:
        print("No grad for", name)
    else:
        gnorm = p.grad.detach().abs().mean().item()
        total_norm += gnorm
        print(f"grad mean {name}: {gnorm:.6e}")
print("Sum of grad means:", total_norm)
opt.step()

# forward again to see change
with torch.no_grad():
    logits2 = model(frames, text_data).view(-1)
    loss2 = criterion(logits2, labels)
print("Loss after one step:", float(loss2.item()))
print("logits diff (mean):", float((logits2 - logits).mean().item()))
