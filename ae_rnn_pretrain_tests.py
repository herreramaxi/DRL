# ae_pretrain.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from ae_rnn_pretrain import SimpleLSTMAutoencoder

# Testing code
def test_autoencoder_shapes():
    seq_len = 8
    batch_size = 2
    # Create a random batch of board sequences
    x = torch.randn(batch_size, seq_len, 5, 5, dtype=torch.float32)
    ae = SimpleLSTMAutoencoder(seq_len=seq_len)
    z, recon = ae(x)
    assert z.shape == (batch_size, ae.to_latent.out_features), f"Expected latent shape {(batch_size, ae.to_latent.out_features)}, got {z.shape}"
    assert recon.shape == (batch_size, seq_len, 5, 5), f"Expected recon shape {(batch_size, seq_len, 5, 5)}, got {recon.shape}"
    print("Shape test passed.")

def test_autoencoder_identity():
    # Testing that zero input leads to zero output (with linear layers, not guaranteed exactly zero but close)
    seq_len = 8
    batch_size = 2
    x = torch.zeros(batch_size, seq_len, 5, 5, dtype=torch.float32)
    ae = SimpleLSTMAutoencoder(seq_len=seq_len)
    z, recon = ae(x)
    # recon should be close to zeros
    max_abs = recon.abs().max().item()
    print(f"Max absolute reconstruction value for zero input: {max_abs:.6f} (should be small)")

def test_autoencoder_gradient_flow():
    seq_len = 8
    batch_size = 2
    x = torch.randn(batch_size, seq_len, 5, 5, requires_grad=True)
    ae = SimpleLSTMAutoencoder(seq_len=seq_len)
    z, recon = ae(x)
    loss = nn.MSELoss()(recon, x)
    loss.backward()
    # Ensure gradients exist
    grad_norm = x.grad.norm().item()
    print(f"Gradient norm w.r.t input: {grad_norm:.6f} (should be > 0)")

import pandas as pd

if __name__ == "__main__":
    # Pretrain AE (will regenerate boards.npy if shape is off)
    # pretrain_ae()
    test_autoencoder_shapes()
    test_autoencoder_identity()
    test_autoencoder_gradient_flow()
        
    # Load pre-trained AE
    SEQ_LEN = 8
    device = torch.device("cpu")
    ae = SimpleLSTMAutoencoder(seq_len=SEQ_LEN, board_size=5, hidden_dim=64, latent_dim=8).to(device)
    ae.load_state_dict(torch.load("ae_rnn_pretrained.pth", map_location=device))
    ae.eval()

    # Load sequences
    seqs = np.load("board_seqs.npy")  # shape (N, SEQ_LEN, 5,5)
    # Normalize as during training
    seqs = (seqs + 60000.0) / 120000.0

    # Take first batch of sequences
    batch = torch.from_numpy(seqs[:4]).to(device)  # first 4 sequences
    z, recon = ae(batch)  # z: (4, latent_dim), recon: (4, SEQ_LEN, 5,5)

    # Plot original vs reconstruction for the first sequence
    orig = seqs[0]  # (SEQ_LEN,5,5)
    rec = recon[0].detach().cpu().numpy()

    fig, axes = plt.subplots(2, SEQ_LEN, figsize=(SEQ_LEN*2, 4))
    for i in range(SEQ_LEN):
        axes[0, i].imshow(orig[i])
        axes[0, i].set_title(f"Orig {i}")
        axes[0, i].axis("off")
        axes[1, i].imshow(rec[i])
        axes[1, i].set_title(f"Recon {i}")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.show()

    # Display latent vector for the first sequence
    latent = z[0].detach().cpu().numpy()
    df = pd.DataFrame(latent.reshape(1, -1), columns=[f"z{i}" for i in range(latent.shape[0])])
    print("Latent vector for sequence 0:")
    print(df)
    