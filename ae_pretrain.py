# ae_pretrain.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common import get_device_name, make_env_masking_enabled, parse_arguments_ae
import gymnasium as gym
from ChessGame.ChessEnv import register_chess_env

# ─── Settings ─────────────────────────────────────
# N_SAMPLES   = 50_000
BATCH_SIZE  = 512
AE_LR       = 1e-3
# AE_EPOCHS   = 20
INPUT_DIM   = 5 * 5
HIDDEN_DIM  = 64
LATENT_DIM  = 8
# SAVE_BOARDS = "boards.npy"
# SAVE_AE     = "ae_pretrained.pth"
# ──────────────────────────────────────────────────

register_chess_env()

def collect_boards(args):    
    env = make_env_masking_enabled(True)
    obs, _ = env.reset()

    boards = []
    pbar = tqdm(total=args.n_samples, desc="Collecting boards")
    while len(boards) < args.n_samples:
        # 1) get legal moves set, convert to list
        legal_moves = list(env.unwrapped.get_legal_moves())
        # 2) sample one valid action
        action = int(np.random.choice(legal_moves))
        # 3) step
        obs, _, terminated, truncated, _ = env.step(action)
        # 4) store **full** 5×5 board
        boards.append(obs["board"].copy())
        pbar.update(1)
        # 5) reset if done
        if terminated or truncated:
            obs, _ = env.reset()
    pbar.close()

    boards = np.stack(boards).astype(np.float32)  # shape (N_SAMPLES,5,5)
    np.save(args.board_file_path, boards)
    print(f"Saved {boards.shape[0]} boards --> {args.board_file_path}")

class BoardAutoencoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),                          # (B,5,5)-->(B,25)
            nn.Linear(input_dim, hidden_dim),      # (B,25)-->(B,64)
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),     # (B,64)-->(B,8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),     # (B,8)-->(B,64)
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),      # (B,64)-->(B,25)
            nn.Sigmoid(),                          # (B,25)-->[0,1]
            nn.Unflatten(1, (5, 5)),               # -->(B,5,5)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

def pretrain_ae(args):
    # 1) load & normalize boards
    boards = np.load(args.board_file_path)   
    boards = boards.astype(np.float32)
    boards = (boards + 60000.0) / 120000.0

    # 2) dataset & loader
    loader = DataLoader(torch.from_numpy(boards), batch_size=BATCH_SIZE, shuffle=True)
    # for batch in loader:  # now batch is (B,5,5) directly

    # dataset = TensorDataset(torch.from_numpy(boards))
    # loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3) model & optimizer
    device = get_device_name()
    ae = BoardAutoencoder().to(device)
    optimizer = optim.Adam(ae.parameters(), lr=AE_LR)
    loss_fn = nn.MSELoss()

    # 4) training loop
    for epoch in range(1, args.n_epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)         # (B,5,5)
            _, recon = ae(batch)             # recon: (B,5,5)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg = total_loss / len(loader.dataset)
        print(f"Epoch {epoch:2d}/{args.n_epochs}  AE loss = {avg:.6f}")

    # 5) save weights
    torch.save(ae.state_dict(), args.weights_file_path)
    print(f"Saved pretrained AE --> {args.weights_file_path}")

if __name__ == "__main__":
    args = parse_arguments_ae(50_000,"boards/boards.npy","weights/ae_pretrained.pth")   

    if args.force_clean == "True":
        if os.path.exists(args.weights_file_path):
            os.remove(args.weights_file_path)
            print(f"Deleted {args.weights_file_path}")
        if os.path.exists(args.board_file_path):
            os.remove(args.board_file_path)
            print(f"Deleted {args.board_file_path}")

    if not os.path.exists(args.weights_file_path):
        print("Generating boards for autoencoder")
        collect_boards(args)
        print("Training autoencoder")
        pretrain_ae(args)
    else:
        print("Skipping autoencoder training, weights already exists")
