from torch.distributions.categorical import Categorical
from torch.distributions.normal import  Normal
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from tensorboardX import SummaryWriter
import random


def run_step(env,s, agent):
    a = agent.get_action([s])
    next_s,r,done, _ = env.step(a)
    if done:
        next_s = env.reset()
    return Transition(s,a,r,next_s,done)


class PGAgent():
    def __init__(self,model,device):
        """A simple PG agent"""
        self.model = model
        self.device = device

    def get_action(self, state):
        """interface for Agent"""
        s = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mu, var = self.model(s)
        m = Normal(mu, var)
        return m.sample().cpu().data.numpy()[0].clip(-1, 1)


class ActorNet(nn.Module):
    def __init__(self, n_in, n_out, hid1_size=128, hid2_size=64):
        super(ActorNet, self).__init__()
        self.layer1 = nn.Linear(n_in, hid1_size)
        self.layer2 = nn.Linear(hid1_size, hid2_size)
        self.mu = nn.Linear(hid2_size, n_out)
        self.var = nn.Linear(hid2_size, n_out)

    def forward(self, x_in):
        hidden = F.relu(self.layer1(x_in))
        hidden = F.relu(self.layer2(hidden))
        return torch.tanh(self.mu(hidden)), F.softplus(self.var(hidden))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def head(self):
        return self.memory[-1]

    def __len__(self):
        return len(self.memory)


def evaluate(env, agent, n_games=1):
    """ Plays n_games full games, Returns mean reward. """
    t_max = env.spec.timestep_limit or 1000
    rewards = []

    for _ in range(n_games):
        s = env.reset()
        reward = 0.0
        for _ in range(t_max):
            action = agent.get_action(np.array([s]))
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)

    return np.mean(rewards)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# for training
STEPS = 20000
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
ENV_SIZE = 32
BETA = 0.001
GAMMA = 1.0
TAU = 0.02
# GAME = "LunarLander-v2"
GAME = 'BipedalWalker-v2'


batch_envs = [gym.make(GAME) for _ in range(ENV_SIZE)]
# for evaluation
eval_env = gym.make(GAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x, y = eval_env.observation_space.shape[0], eval_env.action_space.shape[0]

actor = ActorNet(x, y).to(device)
critic = torch.nn.Sequential(nn.Linear(x, 128), nn.ReLU(),nn.Linear(128, 1)).to(device)
critic_target = torch.nn.Sequential(nn.Linear(x, 128),nn.ReLU(), nn.Linear(128, 1)).to(device)
critic_target.load_state_dict(critic.state_dict())

agent = PGAgent(actor, device)

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)

eval_env.seed(0)
torch.random.manual_seed(0)

states = [env.reset() for env in batch_envs]
writer = SummaryWriter(comment="a2c_continuous")

replay_buffer = ReplayMemory(20000)

for step_idx in range(STEPS):

    batch_transitions = [run_step(env, s, agent) for env, s in zip(batch_envs, states)]
    states = [t.next_state for t in batch_transitions]

    #[replay_buffer.push(transition) for transition in batch_transitions]
    #if len(replay_buffer) < BATCH_SIZE:
    #    continue

    #batch_transitions.extend(replay_buffer.sample(BATCH_SIZE-ENV_SIZE))

    batch = Transition(*zip(*batch_transitions))

    states_t = torch.FloatTensor(batch.state).to(device)
    actions_t = torch.FloatTensor(batch.action).to(device)
    next_states_t = torch.FloatTensor(batch.next_state).to(device)
    rewards_t = torch.FloatTensor(batch.reward).to(device)
    done_t = torch.FloatTensor(batch.done).to(device)

    # critic loss
    predicted_states_v = critic(states_t).squeeze()
    with torch.no_grad():
        next_states_v = critic_target(next_states_t).squeeze() * (1 - done_t)
        target_states_v = next_states_v * GAMMA + rewards_t

    L_critic = F.smooth_l1_loss(predicted_states_v, target_states_v)

    optimizer_critic.zero_grad()
    L_critic.backward()
    optimizer_critic.step()

    # actor loss
    mu, var = actor(states_t)
    m = Normal(mu, var)
    log_probs_t = m.log_prob(actions_t)
    #log_probs_t = log_probs_t.clamp(min=-10)
    advantages_t = (target_states_v - predicted_states_v).detach()
    J_actor = (advantages_t.unsqueeze(-1) * log_probs_t).mean()


    # entropy
    entropy = m.entropy().mean()
    L_actor = -J_actor - entropy * BETA

    optimizer_actor.zero_grad()
    L_actor.backward()
    optimizer_actor.step()

    # smooth update target
    for target_param, new_param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data = target_param.data * (1 - TAU) + new_param.data * TAU


    writer.add_scalar("Entropy", entropy, step_idx)
    writer.add_scalar("Critic_Loss", L_critic, step_idx)
    writer.add_scalar("Actor_Loss", L_actor, step_idx)
    writer.add_scalar("V",predicted_states_v.mean(),step_idx)

    if step_idx % 50 == 0:
        score = evaluate(eval_env, agent, n_games=5)
        print("Step {}: with score : {:.3f}".format(step_idx, score))
        writer.add_scalar("Score", score, step_idx)

        if score>= 300:
            print("Reach the target score 300 of 5 games")
            break

