# debug_probs.py
import numpy as np
import torch
from models.hybrid_model import create_model, get_model_config
from datasets.deepfake_dataset import create_data_loaders, get_dataset_config
from config import config

cfg = get_dataset_config()
cfg['batch_size'] = 4
cfg['num_workers'] = 0

train_loader, _ = create_data_loaders(data_dir=str(config.DATA_DIR / "processed"), **cfg)
frames, text_data, labels, meta = next(iter(train_loader))

model = create_model(get_model_config())
model.eval()
with torch.no_grad():
    logits = model(frames, text_data).view(-1)
    probs = torch.sigmoid(logits)

labels = labels.float().view(-1)

print("labels unique/counts:", np.unique(labels.numpy(), return_counts=True))
print("sample labels:", labels.numpy()[:10])
print("logits sample:", logits.detach().cpu().numpy()[:10])
print("probs sample:", probs.detach().cpu().numpy()[:10])
print("probs stats: min, max, mean, std:", float(probs.min()), float(probs.max()), float(probs.mean()), float(probs.std()))
print("unique probs (rounded):", np.unique(np.round(probs.detach().cpu().numpy(), 4))[:20])
