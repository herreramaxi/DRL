# Enhancing the learning capabilities of Deep Reinforcement Learning agents in chess with Recurrent Neural Network (RNN)-based autoencoders

This project explores how RNN-based Autoencoders can enhance Deep Reinforcement Learning (DRL) agents in chess using a reduced 5x5 variant called Gardner.
Chess has long been a benchmark for AI progress, from Deep Blue and Stockfish to AlphaZero. While DRL achieves superhuman play, it remains computationally expensive and slow to train.
Eight DRL agents were trained against a random opponent, including PPO and Maskable PPO (baseline), RNN-based PPO, Autoencoder PPO (Feedforward and LSTM) in recurrent and non-recurrent configurations, and Transformer PPO using GPT-2 as the encoder.

Key results:
* After 1M steps
    * Transformer PPO reached a 77.4% win rate (+13.2pp).
    * Feedforward Autoencoder Maskable Recurrent PPO reached a 77.0% win rate (+12.8pp).
    * LSTM Autoencoder Maskable Recurrent PPO reached a 76.9% win rate (+12.7pp).
    * Maskable PPO (Baseline) reached a 64.20% win rate. 
* After 5M steps, all agents converged to ~96% win rate vs. the random opponent.

Conclusion: RNN-based autoencoders improve short-term learning speed but increase complexity and compute requirements. In the long run, all agents easily beat random opponents.

## Conda Environment Setup
### Install and Activate Conda Environment
1. conda env create --file=requirements.yml  
2. conda activate gymnasium-env

### Deactivate Environment
conda deactivate

#### Update Enviroment
conda env update --file=requirements.yml  

## Running Experiments

### Experiment 1 (1M steps)
nohup python [train.py](https://github.com/herreramaxi/DRL/blob/main/train.py) \
    --num-repeats 3 \
    --n-envs 16 \
    --total-timesteps 1000000 \
    --batch-size 16384 \
    --n-steps 4096 \
    --n-epochs 10 \
    --share-features-extractor \
    > logs.out 2>&1 &

### Experiment 2 (5M steps)
nohup python [train.py](https://github.com/herreramaxi/DRL/blob/main/train.py) \
    --num-repeats 3 \
    --n-envs 10 \
    --total-timesteps 5000000 \
    --batch-size 8192 \
    --n-steps 4096 \
    --n-epochs 10 \
    --share-features-extractor \
    --experiments 2 5 7 8 \
    > logs.out 2>&1 &

### Basic Run
python [train.py](https://github.com/herreramaxi/DRL/blob/main/train.py) --total-timesteps 100000 --n-envs 4 --num-repeats 1 --share-features-extractor

### Examples
python [train.py](https://github.com/herreramaxi/DRL/blob/main/train.py) --total-timesteps 1000000 --n-envs 16 --batch-size 16384 --n-steps 4096 --n-epochs 10 --num-repeats 3 --share-features-extractor
python [train.py](https://github.com/herreramaxi/DRL/blob/main/train.py) --parallel True --max-workers 4 --total-timesteps 1000000 --n-envs 4 --n-steps 4096 --batch-size 4096 --num-repeats 3

## Evaluate Model

## Evaluate Model (1M steps) 
* python [evaluate_agent.py](https://github.com/herreramaxi/DRL/blob/main/evaluate_agent.py) --rl-algorithm MaskablePPO --model-path ./models/8_Naive_Transformer_PPO_20250805_011256
* python [evaluate_agent.py](https://github.com/herreramaxi/DRL/blob/main/evaluate_agent.py) --rl-algorithm MaskableRecurrentPPO --model-path ./models/7_LSTM_Autoencoder_MaskableRecurrentPPO_20250805_005518

## Evaluate Model (5M steps)
* python [evaluate_agent.py](https://github.com/herreramaxi/DRL/blob/main/evaluate_agent.py) --rl-algorithm MaskablePPO --model-path ./models/8_Naive_Transformer_PPO_20250806_221310
* python [evaluate_agent.py](https://github.com/herreramaxi/DRL/blob/main/evaluate_agent.py) --rl-algorithm MaskableRecurrentPPO --model-path ./models/7_LSTM_Autoencoder_MaskableRecurrentPPO_20250806_194219

## GPU Monitoring
nvidia-smi

## Implemented Agents
* [Chess_1_PPO.py](https://github.com/herreramaxi/DRL/blob/main/Chess_1_PPO.py)
* [Chess_2_MaskablePPO.py](https://github.com/herreramaxi/DRL/blob/main/Chess_2_MaskablePPO.py)
* [Chess_3_MaskableRecurrentPPO.py](https://github.com/herreramaxi/DRL/blob/main/Chess_3_MaskableRecurrentPPO.py)
* [Chess_4_FF_AutoEncoder_MaskablePPO.py](https://github.com/herreramaxi/DRL/blob/main/Chess_4_FF_AutoEncoder_MaskablePPO.py)
* [Chess_5_FF_Autoencoder_MaskableRecurrentPPO.py](https://github.com/herreramaxi/DRL/blob/main/Chess_5_FF_Autoencoder_MaskableRecurrentPPO.py)
* [Chess_6_LSTM_Autoencoder_MaskablePPO.py](https://github.com/herreramaxi/DRL/blob/main/Chess_6_LSTM_Autoencoder_MaskablePPO.py)
* [Chess_7_LSTM_Autoencoder_MaskableRecurrentPPOv.py](https://github.com/herreramaxi/DRL/blob/main/Chess_7_LSTM_Autoencoder_MaskableRecurrentPPO.py)
* [Chess_8_Transformer.py](https://github.com/herreramaxi/DRL/blob/main/Chess_8_Transformer.py)

## Gardner (Customized) 
[ChessEnv.py](https://github.com/herreramaxi/DRL/blob/main/ChessGame/ChessEnv.py)

## References

### Gardner MiniChess
Special thanks to Robert and Michael for providing their consent on using Gardner MiniChess environment.
[shiningsunnyday/mcts-chess/](https://github.com/shiningsunnyday/mcts-chess/)

### MaskableRecurrentPPO
Implementation of maskable recurrent ppo algorithm for contrib package for Stable-Baselines3.
[MaskableRecurrentPPO](https://github.com/akbaig/stable-baselines3-contrib)




