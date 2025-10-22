import os
from contextlib import nullcontext
import numpy as np
import torch
import bdh
import math 
# Config to run on weak hardware
BDH_CONFIG = bdh.MicroBDHConfig() 
BLOCK_SIZE = 256
BATCH_SIZE = 8
MAX_ITERS = 50000

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_ITERS = 20000
LR_DECAY_ITERS = MAX_ITERS
MIN_LR = 3e-5

LOG_FREQ = 1
EVAL_FREQ = 100
GRADIENT_ACCUMULATION_STEPS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype) if "cuda" in device.type else nullcontext()
scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))
torch.manual_seed(1337)
print(f"Using device: {device} with data type {dtype}")

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10)
        for k in range(10):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

model = bdh.BDH(BDH_CONFIG).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
print(f"Model created. Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} million.")

iter_num = 0
best_val_loss = 1e9
if os.path.exists('bdh_checkpoint.pt'):
    print("Checkpoint ‘bdh_checkpoint.pt’ found. Loading to resume...")
    checkpoint = torch.load('bdh_checkpoint.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Loading complete. Resuming from step {iter_num+1}.")

print("\n--- START OF TRAINING ---")
for step in range(iter_num, MAX_ITERS):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if step % EVAL_FREQ == 0 or step == MAX_ITERS - 1:
        losses = estimate_loss(model)
        print(f"Шаг {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            print(f"  -> New best val loss! Saving checkpoint and model for inference.")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter_num': step,
                'best_val_loss': best_val_loss,
                'config': BDH_CONFIG,
            }
            torch.save(checkpoint, 'bdh_checkpoint.pt')
            torch.save(model.state_dict(), 'bdh_micro.pt')


    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(GRADIENT_ACCUMULATION_STEPS):
        x, y = get_batch("train")
        with ctx:
            logits, loss = model(x, y)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        scaler.scale(loss).backward()
    
    scaler.step(optimizer)
    scaler.update()

    if step % LOG_FREQ == 0:
         print(f"Шаг: {step}/{MAX_ITERS}, loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}, lr: {lr:.6f}")

print("\n--- TRAINING COMPLETED ---")
model.load_state_dict(torch.load('bdh_micro.pt', map_location=device))
model.eval()
prompt_str = "[USER]What is the capital of France?[ASSISTANT]"
prompt = torch.tensor(bytearray(prompt_str, "utf-8"), dtype=torch.long, device=device).unsqueeze(0)
ret = model.generate(prompt, max_new_tokens=50, top_k=5)
ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(errors="backslashreplace")
print("\n--- GENERATED TEXT ---")
print(ret_decoded)