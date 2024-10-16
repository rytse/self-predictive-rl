import numpy as np
from omegaconf import DictConfig
import utils
from utils import logger
from gym import Env
import torch


def make_agent(env: Env, device: torch.device, cfg: DictConfig):
    if cfg.agent == "alm":
        from agents.alm import AlmAgent

        num_states = np.prod(env.observation_space.shape)
        num_actions = np.prod(env.action_space.shape)
        action_low = env.action_space.low[0]
        action_high = env.action_space.high[0]

        reward_low = env.reward_range[0]
        reward_high = env.reward_range[1]

        if cfg.id == "BipedalWalker-v3":
            reward_low = -200
            reward_high = 400
        elif reward_low == -np.inf or reward_high == np.inf:
            assert not cfg.bisim_norm_reward

        if cfg.id == "Humanoid-v2":
            cfg.env_buffer_size = 1000000
        buffer_size = min(cfg.env_buffer_size, cfg.num_train_steps)

        agent = AlmAgent(
            device,
            action_low,
            action_high,
            reward_low,
            reward_high,
            num_states,
            num_actions,
            buffer_size,
            cfg,
        )

    else:
        raise NotImplementedError

    return agent


def make_env(cfg):
    if cfg.benchmark == "gym":
        import gym

        if cfg.id == "T-Ant-v2" or cfg.id == "T-Humanoid-v2":
            utils.register_mbpo_environments()

        def get_env(cfg):
            env = gym.make(cfg.id)

            if cfg.distraction > 0:
                from workspaces.distracted_env import (
                    GaussianDistractedWrapper,
                    GaussianMixtureDistractedWrapper,
                    InterleavedGaussianMixtureDistractedWrapper,
                )

                if cfg.distraction_type == "gaussian":
                    env = GaussianDistractedWrapper(
                        env,
                        distraction_dims=cfg.distraction,
                        distraction_scale=cfg.scale,
                    )
                elif cfg.distraction_type == "gaussian_mixture":
                    env = GaussianMixtureDistractedWrapper(
                        env,
                        distraction_dims=cfg.distraction,
                        distraction_scale=cfg.scale,
                    )
                elif cfg.distraction_type == "interleaved_gaussian_mixture":
                    env = InterleavedGaussianMixtureDistractedWrapper(
                        env,
                        distraction_dims=cfg.distraction,
                        distraction_scale=cfg.scale,
                    )
                else:
                    raise NotImplementedError

            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed=cfg.seed)
            env.observation_space.seed(cfg.seed)
            env.action_space.seed(cfg.seed)
            logger.log(env.observation_space.shape, env.action_space)
            return env

        return get_env(cfg), get_env(cfg)

    else:
        raise NotImplementedError
