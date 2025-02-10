import collections
import math
from collections import OrderedDict

import hydra
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.ensemble_ddpg import EnsembleDDPGAgent


class CIC(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim, project_skill):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

        self.state_net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.skill_dim),
        )

        self.next_state_net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.skill_dim),
        )

        self.pred_net = nn.Sequential(
            nn.Linear(2 * self.skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.skill_dim),
        )

        if project_skill:
            self.skill_net = nn.Sequential(
                nn.Linear(self.skill_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.skill_dim),
            )
        else:
            self.skill_net = nn.Identity()

        self.apply(utils.weight_init)

    def forward(self, state, next_state, skill):
        assert len(state.size()) == len(next_state.size())
        state = self.state_net(state)
        next_state = self.state_net(next_state)
        query = self.skill_net(skill)
        key = self.pred_net(torch.cat([state, next_state], 1))
        return query, key


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "mps")


class RMS(object):
    def __init__(self, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (
            self.S * self.n
            + torch.var(x, dim=0) * bs
            + (delta**2) * self.n * bs / (self.n + bs)
        ) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class APTArgs:
    def __init__(
        self,
        knn_k=16,
        knn_avg=True,
        rms=True,
        knn_clip=0.0005,
    ):
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.rms = rms
        self.knn_clip = knn_clip


rms = RMS()


def compute_apt_reward(source, target, args):

    b1, b2 = source.size(0), target.size(0)
    # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
    sim_matrix = torch.cdist(source, target, p=2)

    reward, _ = sim_matrix.topk(
        args.knn_k, dim=1, largest=False, sorted=True
    )  # (b1, k)

    if not args.knn_avg:  # only keep k-th nearest neighbor
        reward = reward[:, -1]
        reward = reward.reshape(-1, 1)  # (b1, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(
            reward - args.knn_clip, torch.zeros_like(reward).to(device)
        )  # (b1, )
    else:  # average over all k nearest neighbors
        reward = reward.reshape(-1, 1)  # (b1 * k, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward).to(device))
        reward = reward.reshape((b1, args.knn_k))  # (b1, k)
        reward = reward.mean(dim=1)  # (b1,)
    reward = torch.log(reward + 1.0)
    return reward


class SBRLV4Agent(EnsembleDDPGAgent):
    def __init__(
        self,
        update_skill_every_step,
        skill_dim,
        update_encoder,
        contrastive_update_rate,
        temperature,
        alpha,
        skill,
        project_skill,
        update_rep,
        ensemble_size,
        **kwargs
    ):
        self.skill_dim = skill_dim
        self.ensemble_size = ensemble_size
        self.update_skill_every_step = update_skill_every_step
        self.update_encoder = update_encoder
        self.contrastive_update_rate = contrastive_update_rate
        self.temperature = temperature
        self.alpha = alpha
        # specify skill in fine-tuning stage if needed
        self.skill = int(skill) if skill >= 0 else np.random.choice(self.skill_dim)
        self.update_rep = update_rep
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim
        kwargs["ensemble_size"] = ensemble_size
        self.batch_size = kwargs["batch_size"]
        # create actor and critic
        super().__init__(**kwargs)

        self.cic = CIC(
            self.obs_dim, skill_dim, kwargs["hidden_dim"], project_skill
        ).to(kwargs["device"])

            # optimizers
        self.cic_optimizer = torch.optim.Adam(self.cic.parameters(), lr=self.lr)

        self.cic.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, "skill"),)

    def init_meta(self):
        skill = np.zeros(self.skill_dim).astype(np.float32)
        if not self.reward_free:
            skill[self.skill] = 1.0
        else:
            skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta["skill"] = skill
        return meta

    def create_mask_matrix(self, cluster_index):
        matrix = np.zeros((self.ensemble_size, int(self.batch_size)), dtype=np.float32)
        matrix[cluster_index] = 1.0

        return torch.from_numpy(matrix).unsqueeze(-1).to(self.device)

    def update_meta(self, meta, global_step, time_step, finetune=False):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def compute_cpc_loss(self, obs, next_obs, skill):
        temperature = self.temperature
        eps = 1e-6
        query, key = self.cic.forward(obs, next_obs, skill)
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        cov = torch.mm(query, key.T)  # (b,b)
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)  # (b,)
        row_sub = (
            torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        )
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = torch.exp(torch.sum(query * key, dim=-1) / temperature)  # (b,)
        loss = -torch.log(pos / (neg + eps))  # (b,)
        return loss, cov / temperature

    def update_cic(self, obs, skill, next_obs, step):
        metrics = dict()

        loss, logits = self.compute_cpc_loss(obs, next_obs, skill)
        loss = loss.mean()
        self.cic_optimizer.zero_grad()
        loss.backward()
        self.cic_optimizer.step()

        if self.use_tb or self.use_wandb:
            metrics["cic_loss"] = loss.item()
            metrics["cic_logits"] = logits.norm()

        return metrics

    @torch.no_grad()
    def compute_apt_reward(self, obs, next_obs):
        args = APTArgs()
        source = self.cic.state_net(obs)
        target = self.cic.state_net(next_obs)
        reward = compute_apt_reward(source, target, args)  # (b,)
        return reward.unsqueeze(-1)  # (b,1)

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device
        )

        with torch.no_grad():
            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(self.update_cic(obs, skill, next_obs, step))

            with torch.no_grad():
                apt_reward = self.compute_apt_reward(next_obs, next_obs)

            if self.use_tb or self.use_wandb:
                metrics["apt_reward"] = apt_reward.mean().item()

            reward = apt_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        # obs = torch.cat([obs, skill], dim=1)
        # next_obs = torch.cat([next_obs, skill], dim=1)

        mask = self.create_mask_matrix(self.skill)

        # update critic
        metrics.update(
            self.update_critic(
                obs.detach(),
                skill,
                action,
                reward,
                discount,
                next_obs.detach(),
                step,
                mask,
            )
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), skill, step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
