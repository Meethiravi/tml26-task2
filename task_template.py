import os
import glob
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file
from scipy.stats import zscore
from tqdm import tqdm

# configuration
NORM_MEAN = (0.5071, 0.4867, 0.4408)
NORM_STD = (0.2675, 0.2565, 0.2761)


# biased_crop logits
def apply_biased_crop(x):
    return transforms.functional.crop(x, top=3, left=6, height=32, width=32)

# model definition
def make_model(num_classes: int = 100) -> nn.Module:
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# data loading and preprocessing
def build_loaders(data_root, train_idx_path, batch_size, num_workers):
    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    
    # standard test set loader

    test_set = datasets.CIFAR100(root=data_root, train=False, download=True, transform=base_tf)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True) 

    biased_tf = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Lambda(apply_biased_crop),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    # biased crop test set loader

    biased_set = datasets.CIFAR100(root=data_root, train=False, download=True, transform=biased_tf)
    biased_loader = DataLoader(biased_set, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, pin_memory=True)

    train_full = datasets.CIFAR100(root=data_root, train=True, download=True, transform=base_tf)
    with open(train_idx_path) as f:
        train_idx = json.load(f)
    if isinstance(train_idx, dict):
        for k in ("train_idx", "indices", "idx"):
            if k in train_idx: train_idx = train_idx[k]; break
    
    # we have used 10k samples 
    train_idx_sub = list(map(int, train_idx))[:10000] 
    train_loader = DataLoader(Subset(train_full, train_idx_sub), batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_loader, biased_loader, train_loader

# running inference to get raw logits

@torch.no_grad()
def forward_logits(model, loader, device):
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True) # Added non_blocking
        out.append(model(x).cpu())
    return torch.cat(out, dim=0)

# Signals Computation

# layer_wise weight cosine 
def layer_wise_cosine(state_a, state_b):
    scores = []
    for k in state_a:
        if k in state_b and state_a[k].shape == state_b[k].shape:
            a, b = state_a[k].float().view(-1), state_b[k].float().view(-1)
            denom = a.norm() * b.norm()
            if denom > 0: scores.append(float((a @ b) / denom))
    return np.mean(scores) if scores else 0.0

# batch norm stats (running mean, running var)
def bn_stat_similarity(state_a, state_b):
    scores = []
    for k in state_a:
        if ('running_mean' in k or 'running_var' in k) and k in state_b:
            a, b = state_a[k].float().view(-1), state_b[k].float().view(-1)
            denom = a.norm() * b.norm()
            if denom > 0: scores.append(float((a @ b) / denom))
    return np.mean(scores) if scores else 0.0

# pearson correlation 
def pearson_per_row(A, B):
    A = A - A.mean(dim=1, keepdim=True)
    B = B - B.mean(dim=1, keepdim=True)
    return ( (A * B).sum(dim=1) / (A.norm(dim=1) * B.norm(dim=1) + 1e-12) )

# coimcident error agreement
def coincident_error_agreement(tgt_logits, sus_logits, true_labels):
    tgt_preds, sus_preds = tgt_logits.argmax(1), sus_logits.argmax(1)
    wrong_mask = (tgt_preds != true_labels)
    if not wrong_mask.any(): return 0.0
    return (sus_preds[wrong_mask] == tgt_preds[wrong_mask]).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_dir", required=True)
    ap.add_argument("--data_root", default="./cifar100_data")
    ap.add_argument("--out_csv", default="./submission.csv")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=8)   
    ap.add_argument("--save_raw", default="./raw_signals.npz")
    args = ap.parse_args()

    # device setup

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # setting up paths for checkpoints and indices

    target_ckpt = os.path.join(args.repo_dir, "target_model", "weights.safetensors")
    train_idx_path = os.path.join(args.repo_dir, "target_model", "train_main_idx.json")
    suspect_files = sorted(glob.glob(os.path.join(args.repo_dir, "suspect_models", "*.safetensors")))

    print("\n[1/3] Preparing Target and Data...")
    target_state = load_file(target_ckpt, device="cpu")
    target = make_model().to(device)
    target.load_state_dict(target_state)
    target.eval()

    # build data loaders

    test_l, biased_l, train_l = build_loaders(args.data_root, train_idx_path, args.batch_size, args.num_workers)
    true_labels = torch.tensor(test_l.dataset.targets)

    # get target reference outputs

    tgt_test_logits = forward_logits(target, test_l, device)
    tgt_biased_logits = forward_logits(target, biased_l, device)
    tgt_train_preds = forward_logits(target, train_l, device).argmax(1)

    # processing all suspect models

    N = len(suspect_files)
    signals = {k: np.zeros(N) for k in ['W', 'BN', 'L', 'LB', 'T', 'E']}
    model = make_model().to(device)

    print(f"\n[2/3] Processing {N} suspects...")
    for i, ckpt in enumerate(tqdm(suspect_files, desc="Inference")):
        s_state = load_file(ckpt, device="cpu")
        signals['W'][i] = layer_wise_cosine(target_state, s_state)
        signals['BN'][i] = bn_stat_similarity(target_state, s_state)

        model.load_state_dict(s_state, strict=False)
        model.eval()

        s_test_logits = forward_logits(model, test_l, device)
        s_biased_logits = forward_logits(model, biased_l, device)
        s_train_preds = forward_logits(model, train_l, device).argmax(1)

        signals['L'][i] = pearson_per_row(tgt_test_logits, s_test_logits).mean().item()
        signals['LB'][i] = pearson_per_row(tgt_biased_logits, s_biased_logits).mean().item()
        signals['T'][i] = (s_train_preds == tgt_train_preds).float().mean().item()
        signals['E'][i] = coincident_error_agreement(tgt_test_logits, s_test_logits, true_labels)

    np.savez(args.save_raw, **signals)

    print("\n[3/3] Finalizing Scores...")

    # weighted combination of z-scored signals
    combined = (0.5 * zscore(signals['W'])) + \
               (1.0 * zscore(signals['BN'])) + \
               (1.0 * zscore(signals['L'])) + \
               (2.0 * zscore(signals['LB'])) + \
               (2.0 * zscore(signals['T'])) + \
               (3.0 * zscore(signals['E']))
    
    # min-max normalization to bring score between 0 and 1
    final_score = (combined - combined.min()) / (combined.max() - combined.min() + 1e-12)

    # save submission file
    pd.DataFrame({"id": np.arange(N), "score": final_score}).to_csv(args.out_csv, index=False)
    print(f"Success! Results saved to {args.out_csv}")

if __name__ == "__main__":
    main()