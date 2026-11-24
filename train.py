# Piece 4: training loop.
# Run from the repo root:  caffeinate -i python train.py
#
# Each epoch:
#   1. train on random 256 x 256 crops from the 252 train chips
#   2. validate on the 89 full-size valid chips -> water IoU
#   3. if this is the best IoU so far, save the weights
#   4. write one line to training_log.csv

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Sen1Floods11Dataset
from unet_model import UNet

# --- Settings (decided together) ---------------------------------------
EPOCHS = 100                    # 2 = smoke test. Change to 100 for the real run.
BATCH_SIZE = 8
CROP_SIZE =256
LEARNING_RATE = 1e-3
CLASS_WEIGHTS = [1.0, 3.0]    # dry, water  (sqrt of the 9.5 : 1 imbalance)
SEED = 42
BASE = 64
RUN_NAME = "run2_base64"
CHECKPOINT = f"checkpoints/{RUN_NAME}_best.pt"
LOG_FILE = f"{RUN_NAME}_log.csv"
# --- Reproducibility: same seed -> same crops, shuffles and start weights.
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Device:", device)

# --- Data ----------------------------------------------------------------
train_set = Sen1Floods11Dataset("train", crop_size=CROP_SIZE)
valid_set = Sen1Floods11Dataset("valid", crop_size=None)
# drop_last=True: skip the leftover batch of 4, so BatchNorm always sees 8.
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=4, shuffle=False, num_workers=0)

# --- Model, loss, optimiser, schedule ----------------------------------
model = UNet(base=BASE).to(device)
# ignore_index=-1: no-data pixels (and the NaN pixels we set to -1) never count.
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, device=device),
                              ignore_index=-1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# Cosine: the learning rate glides from 0.001 down to ~0 over all epochs.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

os.makedirs("checkpoints", exist_ok=True)
with open(LOG_FILE, "w") as f:
    f.write("epoch,train_loss,val_loss,val_iou,lr,seconds\n")

best_iou = -1.0
best_epoch = 0
epoch_times = []

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    lr = optimizer.param_groups[0]["lr"]

    # ---------------- 1. Train ----------------
    model.train()                     # BatchNorm uses batch statistics
    loss_sum = torch.zeros((), device=device)
    n_batches = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        if (labels != -1).sum() == 0:
            continue                  # all no-data: loss would be 0/0 = NaN
        optimizer.zero_grad()         # clear last batch's gradients
        loss = loss_fn(model(images), labels)
        loss.backward()               # work out how each weight should change
        optimizer.step()              # change them
        # .detach() keeps the number without the gradient history.
        # Summing on the GPU avoids waiting for it after every batch.
        loss_sum += loss.detach()
        n_batches += 1
    train_loss = loss_sum.item() / max(n_batches, 1)

    # ---------------- 2. Validate ----------------
    model.eval()                      # BatchNorm uses its learned averages
    val_loss_sum = 0.0
    val_batches = 0
    tp = fp = fn = 0
    with torch.no_grad():             # no gradients needed: faster, less memory
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if (labels != -1).sum() > 0:
                val_loss_sum += loss_fn(logits, labels).item()
                val_batches += 1
            preds = logits.argmax(dim=1)          # 0 = dry, 1 = water
            # labels == 0 or == 1 already excludes the -1 pixels.
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    val_loss = val_loss_sum / max(val_batches, 1)
    val_iou = tp / max(tp + fp + fn, 1)

    scheduler.step()                  # lower the learning rate for next epoch

    # ---------------- 3. Save best ----------------
    marker = ""
    if val_iou > best_iou:
        best_iou, best_epoch = val_iou, epoch
        torch.save(model.state_dict(), CHECKPOINT)
        marker = "  <- best, saved"

    # ---------------- 4. Log ----------------
    seconds = time.time() - start
    epoch_times.append(seconds)
    print(f"Epoch {epoch:3d}/{EPOCHS}  train_loss {train_loss:.4f}  "
          f"val_loss {val_loss:.4f}  val_IoU {val_iou:.4f}  "
          f"lr {lr:.6f}  {seconds:.1f}s{marker}")
    with open(LOG_FILE, "a") as f:
        f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_iou:.6f},"
                f"{lr:.8f},{seconds:.2f}\n")

avg = sum(epoch_times) / len(epoch_times)
print(f"\nBest val IoU {best_iou:.4f} at epoch {best_epoch}  -> {CHECKPOINT}")
print(f"Average {avg:.1f} s per epoch  ->  100 epochs ~ {avg * 100 / 60:.0f} min")
