from typing import Tuple
import random
import torch
import time
import numpy as np
import numpy.typing as npt

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from pathlib import Path
from utils.env import save_frames_as_gif
from utils import logger
from workspaces.common import make_agent, make_env


class MujocoWorkspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        if self.cfg.save_snapshot:
            self.checkpoint_path = self.work_dir / "checkpoints"
            self.checkpoint_path.mkdir(exist_ok=True)

        if "cuda" in cfg.device and not torch.cuda.is_available():
            raise ValueError(
                "Run config specifies a cuda device, but no GPU is available."
            )

        self.device = torch.device(cfg.device)
        self.set_seed()
        self.train_env, self.eval_env = make_env(self.cfg)
        self.agent = make_agent(self.train_env, self.device, self.cfg)
        self.verbose = cfg.verbose  # 0: no print, 1: eval print, 2: train + eval print
        self._train_step = 0
        self._train_episode = 0
        self._best_eval_returns = -np.inf

    def set_seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

    def train(self):
        progress_bar = Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )

        self._explore()
        self._eval()

        state, done, episode_start_time = self.train_env.reset(), False, time.time()
        training_iterator = range(
            1, self.cfg.num_train_steps - self.cfg.explore_steps + 1
        )

        if not self.cfg.debug:
            with progress_bar as p:
                for _ in p.track(training_iterator):
                    state, episode_start_time = self._train_iter(
                        state, episode_start_time
                    )
        else:
            for _ in training_iterator:
                state, episode_start_time = self._train_iter(state, episode_start_time)

        self.train_env.close()

    def _train_iter(
        self, state: npt.NDArray, episode_start_time: float
    ) -> Tuple[npt.NDArray, float]:
        action = self.agent.get_action(state, self._train_step)
        next_state, reward, done, info = self.train_env.step(action)
        self._train_step += 1

        self.agent.env_buffer.push(
            (
                state,
                action,
                reward,
                next_state,
                False if info.get("TimeLimit.truncated", False) else done,
            )
        )

        self.agent.update(self._train_step)

        if (self._train_step) % self.cfg.eval_episode_interval == 0:
            self._eval()

        if (
            self.cfg.save_snapshot
            and (self._train_step) % self.cfg.save_snapshot_interval == 0
        ):
            self.save_snapshot()

        if done:
            self._train_episode += 1
            if self.verbose >= 2:
                print(
                    "TRAIN Episode: {}, total numsteps: {}, return: {}".format(
                        self._train_episode,
                        self._train_step,
                        round(info["episode"]["r"], 2),
                    )
                )
            episode_metrics = dict()
            episode_metrics["train/length"] = info["episode"]["l"]
            episode_metrics["train/return"] = info["episode"]["r"]
            episode_metrics["FPS"] = info["episode"]["l"] / (
                time.time() - episode_start_time
            )
            # episode_metrics["env_buffer_length"] = len(self.agent.env_buffer)
            logger.record_step("env_steps", self._train_step)
            for k, v in episode_metrics.items():
                logger.record_tabular(k, v)
            logger.dump_tabular()

            state, done, episode_start_time = (
                self.train_env.reset(),
                False,
                time.time(),
            )
        else:
            state = next_state

        return state, episode_start_time

    def _explore(self):
        state, done = self.train_env.reset(), False

        for _ in range(1, self.cfg.explore_steps):
            action = self.train_env.action_space.sample()
            next_state, reward, done, info = self.train_env.step(action)
            self.agent.env_buffer.push(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    False if info.get("TimeLimit.truncated", False) else done,
                )
            )

            if done:
                state, done = self.train_env.reset(), False
            else:
                state = next_state

    def _eval(self):
        returns = 0
        steps = 0
        for _ in range(self.cfg.num_eval_episodes):
            done = False
            state = self.eval_env.reset()
            while not done:
                action = self.agent.get_action(state, self._train_step, eval=True)
                next_state, _, done, info = self.eval_env.step(action)
                state = next_state

            returns += info["episode"]["r"]
            steps += info["episode"]["l"]

            if self.verbose >= 1:
                print(
                    "EVAL Episode: {}, total numsteps: {}, return: {}".format(
                        self._train_episode,
                        self._train_step,
                        round(info["episode"]["r"], 2),
                    )
                )

        eval_metrics = dict()
        eval_metrics["return"] = returns / self.cfg.num_eval_episodes
        eval_metrics["length"] = steps / self.cfg.num_eval_episodes

        if (
            self.cfg.save_snapshot
            and returns / self.cfg.num_eval_episodes >= self._best_eval_returns
        ):
            self.save_snapshot(best=True)
            self._best_eval_returns = returns / self.cfg.num_eval_episodes

        logger.record_step("env_steps", self._train_step)
        for k, v in eval_metrics.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()

    def _render_episodes(self, record):
        frames = []
        done = False
        state = self.eval_env.reset()
        while not done:
            action = self.agent.get_action(state, self._train_step, True)
            next_state, _, done, info = self.eval_env.step(action)
            self.eval_env.render()
            state = next_state
        if record:
            save_frames_as_gif(frames)
        print(
            "Episode: {}, episode steps: {}, episode returns: {}".format(
                i, info["episode"]["l"], round(info["episode"]["r"], 2)
            )
        )

    def _eval_bias(self):
        final_mc_list, final_obs_list, final_act_list = self._mc_returns()
        final_mc_norm_list = np.abs(final_mc_list.copy())
        final_mc_norm_list[final_mc_norm_list < 10] = 10

        obs_tensor = torch.FloatTensor(final_obs_list).to(self.device)
        acts_tensor = torch.FloatTensor(final_act_list).to(self.device)
        lower_bound = self.agent.get_lower_bound(obs_tensor, acts_tensor)

        bias = final_mc_list - lower_bound
        normalized_bias_per_state = bias / final_mc_norm_list

        # metrics = dict()
        # metrics["mean_bias"] = np.mean(bias)
        # metrics["std_bias"] = np.std(bias)
        # metrics["mean_normalised_bias"] = np.mean(normalized_bias_per_state)
        # metrics["std_normalised_bias"] = np.std(normalized_bias_per_state)

    def _mc_returns(self):
        final_mc_list = np.zeros(0)
        final_obs_list = []
        final_act_list = []
        n_mc_eval = 1000
        n_mc_cutoff = 350

        while final_mc_list.shape[0] < n_mc_eval:
            o = self.eval_env.reset()
            reward_list, obs_list, act_list = [], [], []
            r, d, ep_ret, ep_len = 0, False, 0, 0

            while not d:
                a = self.agent.get_action(o, self._train_step, True)
                obs_list.append(o)
                act_list.append(a)
                o, r, d, _ = self.eval_env.step(a)
                ep_ret += r
                ep_len += 1
                reward_list.append(r)

            discounted_return_list = np.zeros(ep_len)
            for i_step in range(ep_len - 1, -1, -1):
                if i_step == ep_len - 1:
                    discounted_return_list[i_step] = reward_list[i_step]
                else:
                    discounted_return_list[i_step] = (
                        reward_list[i_step]
                        + self.cfg.gamma * discounted_return_list[i_step + 1]
                    )

            final_mc_list = np.concatenate(
                (final_mc_list, discounted_return_list[:n_mc_cutoff])
            )
            final_obs_list += obs_list[:n_mc_cutoff]
            final_act_list += act_list[:n_mc_cutoff]

        return final_mc_list, np.array(final_obs_list), np.array(final_act_list)

    def save_snapshot(self, best=False):
        if best:
            snapshot = Path(self.checkpoint_path) / "best.pt"
        else:
            snapshot = Path(self.checkpoint_path) / Path(str(self._train_step) + ".pt")
        save_dict = self.agent.get_save_dict()
        torch.save(save_dict, snapshot)
