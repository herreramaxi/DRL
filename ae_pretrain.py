# ae_pretrain.py

import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ChessGame.ChessEnv import register_chess_env
from ChessMaskablePPO import make_env_masking_enabled
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.maskable.utils import get_action_masks
from tqdm import tqdm
# ─── Settings ─────────────────────────────────────
N_SAMPLES   = 50_000
BATCH_SIZE  = 512
AE_LR       = 1e-3
AE_EPOCHS   = 20
INPUT_DIM   = 5 * 5
HIDDEN_DIM  = 64
LATENT_DIM  = 8
SAVE_BOARDS = "boards.npy"
SAVE_AE     = "ae_pretrained.pth"
# ──────────────────────────────────────────────────

def collect_boards():
    register_chess_env( )
    # env = make_vec_env(
    #     make_env_masking_enabled,
    #     n_envs=1,
    #     vec_env_cls=DummyVecEnv
    # )

    env = gym.make("gymnasium_env/ChessGame-v0", invalid_action_masking=True)
    obs, _ = env.reset() 

    boards = []
    # reset() now returns only obs
    infos = None
    pbar = tqdm(total=N_SAMPLES, desc="Collecting boards")

    while len(boards) < N_SAMPLES:
        # 1) extract the mask from obs if present, else from infos
        unwrapped_env = env.unwrapped
        legal = list(unwrapped_env.unwrapped.get_legal_moves())  # returns a Python list of ints
        print(legal)
        print(type(legal))
        # 2) Sample one valid action
        action = int(np.random.choice(legal))
        # action_masks = unwrapped_env.get_legal_moves()
        # a = np.random.choice(action_masks)       

        print(action)
        # 3) step
        obs, _, terminated, truncated, infos = env.step(action)

        # 4) store the new board state
        boards.append(obs["board"][0].copy())
        pbar.update(1)
        # 5) reset if episode ended
        if terminated or truncated:
            obs = env.reset()
            infos = None

    boards = np.stack(boards).astype(np.float32)
    np.save(SAVE_BOARDS, boards)
    print(f"Saved {boards.shape[0]} boards → {SAVE_BOARDS}")


class BoardAutoencoder(nn.Module):
    def __init__(self,
                 input_dim=INPUT_DIM,
                 hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
            nn.Unflatten(1, (5, 5)),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def pretrain_ae():
    # load boards and normalize into [0,1]
    boards = np.load(SAVE_BOARDS).astype(np.float32)
    boards = (boards + 60000.0) / 120000.0

    dataset = TensorDataset(torch.from_numpy(boards))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = BoardAutoencoder().to(device)
    optimizer = optim.Adam(ae.parameters(), lr=AE_LR)
    loss_fn = nn.MSELoss()

    for epoch in range(1, AE_EPOCHS + 1):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            _, recon = ae(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch:2d}/{AE_EPOCHS}  AE loss = {avg_loss:.6f}")

    torch.save(ae.state_dict(), SAVE_AE)
    print(f"Saved pretrained AE → {SAVE_AE}")


if __name__ == "__main__":
    if not os.path.exists(SAVE_BOARDS):
        collect_boards()
    pretrain_ae()
