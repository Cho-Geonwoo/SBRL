from collections import OrderedDict
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils  # utils.weight_init, utils.soft_update_params, utils.to_torch 등이 포함되어 있다고 가정

###############################################################################
# Encoder (DDPG의 Encoder와 동일)
###############################################################################
class SkillEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        # 여기서는 obs_shape가 (channels, height, width)인 픽셀 입력이라고 가정합니다.
        assert len(obs_shape) == 3, "SkillEncoder는 픽셀 입력을 기대합니다."
        self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )
        self.apply(utils.weight_init)

    def forward(self, obs):
        # 픽셀 값 정규화: [0,255] -> [-0.5, 0.5]
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.size(0), -1)
        return h

###############################################################################
# Skill Actor: observation을 받아 skill 선택을 위한 logits를 출력
###############################################################################
class SkillActor(nn.Module):
    def __init__(self, obs_type, obs_dim, skill_dim, feature_dim, hidden_dim):
        """
        obs_type: "pixels" 또는 "states"
        obs_dim: encoder 후의 observation 차원 (또는 state의 차원)
        skill_dim: 선택 가능한 skill의 개수
        feature_dim, hidden_dim: MLP의 차원
        """
        super().__init__()
        # 간단한 trunk (픽셀 입력인 경우 encoder 후의 벡터를 받음)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        layers = []
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        # pixels인 경우 추가 hidden layer (DDPG와 유사하게)
        if obs_type == "pixels":
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, skill_dim))
        self.policy = nn.Sequential(*layers)
        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        logits = self.policy(h)
        return logits

###############################################################################
# Skill Critic: observation을 받아 각 skill에 대한 Q-value를 추정
###############################################################################
class SkillCritic(nn.Module):
    def __init__(self, obs_type, obs_dim, skill_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.q_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, skill_dim),  # 각 skill별 Q-value 출력
        )
        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        q_values = self.q_net(h)  # shape: (batch, skill_dim)
        return q_values

###############################################################################
# SkillSelectorAgent: observation → skill 선택 (one-hot vector)
###############################################################################
class SkillSelectorAgent:
    def __init__(
        self,
        obs_type,
        obs_shape,
        skill_dim,
        device,
        lr,
        feature_dim,
        hidden_dim,
        update_every_steps,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=10000,
    ):
        """
        obs_type: "pixels" 또는 "states"
        obs_shape: observation의 shape (픽셀일 경우 (C,H,W), state일 경우 (state_dim,))
        skill_dim: 선택 가능한 skill 개수 (예, 10이면 10차원 one-hot vector)
        device: torch.device
        lr: 학습률
        feature_dim, hidden_dim: 네트워크 차원
        update_every_steps: 네트워크 업데이트 주기 (replay buffer 업데이트 시)
        epsilon_*: epsilon-greedy exploration schedule
        """
        self.obs_type = obs_type
        self.skill_dim = skill_dim
        self.device = device
        self.lr = lr
        self.update_every_steps = update_every_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0

        # 만약 픽셀 입력이면 encoder 사용, 아니면 identity
        if obs_type == "pixels":
            self.encoder = SkillEncoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim
        else:
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0]

        # actor 및 critic 네트워크 초기화
        self.actor = SkillActor(obs_type, self.obs_dim, skill_dim, feature_dim, hidden_dim).to(device)
        self.critic = SkillCritic(obs_type, self.obs_dim, skill_dim, feature_dim, hidden_dim).to(device)
        self.target_critic = SkillCritic(obs_type, self.obs_dim, skill_dim, feature_dim, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # 픽셀 입력인 경우 encoder도 업데이트
        if obs_type == "pixels":
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        else:
            self.encoder_opt = None

    def get_epsilon(self, step):
        """epsilon decay schedule"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1.0 * step / self.epsilon_decay)
        return epsilon

    def train(self, training=True):
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_opt is not None:
            self.encoder.train(training)

    def eval(self):
        self.train(False)

    def act(self, obs, step, eval_mode=False):
        """
        obs: numpy array (환경으로부터 받은 observation)
        step: 현재 global step (exploration schedule용)
        eval_mode: 평가 시에는 확률적으로 샘플링하지 않고 argmax 선택
        → 반환값: (skill_dim,) one-hot 벡터 (예: [0, 0, 1, 0, ...])
        """
        self.step_count = step
        obs_tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
        # 픽셀 입력인 경우 encoder를 통과시킵니다.
        encoded_obs = self.encoder(obs_tensor)
        logits = self.actor(encoded_obs)  # shape: (1, skill_dim)
        # Categorical 분포 생성
        dist = torch.distributions.Categorical(logits=logits)
        epsilon = self.get_epsilon(step)
        if eval_mode:
            # 평가 시 가장 큰 값을 갖는 skill 선택
            skill_idx = torch.argmax(logits, dim=-1)
        else:
            # epsilon-greedy: 무작위 선택할 확률 epsilon 적용
            if random.random() < epsilon:
                skill_idx = torch.tensor([random.randrange(self.skill_dim)], device=self.device)
            else:
                skill_idx = dist.sample()
        # one-hot 벡터로 변환
        skill_onehot = np.zeros(self.skill_dim).astype(np.float32)
        skill_onehot[skill_idx] = 1.0
        return skill_onehot

    def update(self, replay_iter, step, gamma=0.99, tau=0.005):
        """
        replay_iter: 미니배치를 반환하는 iterator.
        배치의 각 샘플은 (obs, action, reward, discount, next_obs, *meta) 형태로 구성됩니다.
        여기서 meta에는 예를 들어, 저장된 skill (one-hot vector)가 포함됩니다.
        gamma: discount factor
        tau: target 네트워크 soft update 비율
        """
        # 지정한 주기마다 업데이트 수행
        if step % self.update_every_steps != 0:
            return {}

        metrics = {}
        # 배치 샘플링 (배치 구성: obs, action, reward, discount, next_obs, *meta)
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, *meta = utils.to_torch(batch, self.device)
        # meta의 첫 번째 원소가 저장된 skill이라고 가정 (예: one-hot 벡터)
        if len(meta) > 0:
            stored_skill = meta[0]
        else:
            stored_skill = None

        # observation 인코딩 (픽셀 입력인 경우 encoder를 통해 feature vector 얻음)
        obs_enc = self.encoder(obs)
        next_obs_enc = self.encoder(next_obs)

        # --- Critic 업데이트 ---
        q_values = self.critic(obs_enc)  # shape: (batch, skill_dim)
        # 저장된 skill이 있다면, 그 one-hot 벡터에서 argmax로 index를 취함
        if stored_skill is not None:
            skill_idx = stored_skill.argmax(dim=1, keepdim=True)  # shape: (batch, 1)
        else:
            # 저장된 skill 정보가 없으면 actor가 산출한 skill을 사용 (exploration용)
            logits = self.actor(obs_enc)
            dist = torch.distributions.Categorical(logits=logits)
            sampled_skill = dist.sample()  # shape: (batch,)
            skill_idx = sampled_skill.unsqueeze(1)
        current_q = q_values.gather(1, skill_idx)  # (batch, 1)

        with torch.no_grad():
            next_q = self.target_critic(next_obs_enc)  # (batch, skill_dim)
            next_q_max, _ = next_q.max(dim=1, keepdim=True)
            target_q = reward + discount * gamma * next_q_max

        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        metrics["skill_critic_loss"] = critic_loss.item()

        # --- Actor 업데이트 ---
        logits = self.actor(obs_enc)
        dist = torch.distributions.Categorical(logits=logits)
        sampled_skill = dist.sample()  # (batch,)
        log_prob = dist.log_prob(sampled_skill)  # (batch,)
        # critic이 예측한 Q-value (batch, skill_dim)에서 actor가 선택한 skill의 Q-value 취함
        q_values_for_actor = self.critic(obs_enc)
        actor_q = q_values_for_actor.gather(1, sampled_skill.unsqueeze(1)).squeeze(1)
        # policy gradient 방식: log_prob * Q를 최대화 (최소화 문제로 변환)
        actor_loss = - (log_prob * actor_q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        metrics["skill_actor_loss"] = actor_loss.item()

        # --- Target Critic Soft Update ---
        utils.soft_update_params(self.critic, self.target_critic, tau)

        # 만약 encoder를 업데이트하는 경우 (픽셀 입력)엔 encoder optimizer 도 업데이트
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad()
            # (예시로 critic_loss의 gradient를 encoder에도 전파)
            critic_loss.backward()
            self.encoder_opt.step()

        return metrics
