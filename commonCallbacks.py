from stable_baselines3.common.callbacks import BaseCallback

class WinRateCallback(BaseCallback):
    def __init__(self, log_interval=10000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.wins, self.losses, self.draws = 0, 0, 0
        self.invalid_moves = 0
        self.valid_moves = 0
        self.episodes = 0

    def _on_step(self) -> bool:
        # Check infos from vectorized env
        for info in self.locals.get("infos", []):
            if "move" in info:
                if info["move"] == "invalid":
                   self.invalid_moves += 1
                else:
                     self.valid_moves += 1

            if "result" in info:  # Our env provides result
                self.episodes += 1
                if info["result"] == "win":
                    self.wins += 1
                elif info["result"] == "loss":
                    self.losses += 1
                else:
                    self.draws += 1

        if self.num_timesteps % self.log_interval == 0 and self.episodes > 0:
            win_rate = self.wins / self.episodes
            self.logger.record("custom/invalid_moves", self.invalid_moves)
            self.logger.record("custom/valid_moves", self.valid_moves)
            self.logger.record("custom/win_rate", win_rate)
            self.logger.record("custom/episodes", self.episodes)
            self.logger.record("custom/wins", self.wins)
            self.logger.record("custom/losses", self.losses)
            self.logger.record("custom/draws", self.draws)
            # print(f"[Step {self.num_timesteps}] Win rate: {win_rate:.2f}")

        return True