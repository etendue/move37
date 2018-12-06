from torch.distributions.normal import Normal
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from tensorboardX import SummaryWriter
import random
import torch.multiprocessing as mp
import os


class BatchEnv():
    def __init__(self,env_name, n):
        self.env_name = env_name
        self.n = n
        self.envs = [gym.make(env_name) for _ in range(n)]
        self.states = []

    def step(self, actions):
        transitions = []
        for i in range(self.n):
            next_s, r, done, _ = self.envs[i].step(actions[i])
            if done:
                next_s = self.envs[i].reset()

            transitions.append(Transition(self.states[i], actions[i], r, next_s, done))
            self.states[i] = next_s
        return transitions

    def reset(self):
        self.states = [env.reset() for env in self.envs]


class PGAgent():
    def __init__(self, model, device):
        """A simple PG agent"""
        self.model = model
        self.device = device

    def get_action(self, state):
        """interface for Agent"""
        s = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mu, var = self.model(s)
        m = Normal(mu, var)
        return m.sample().cpu().data.numpy().clip(-1, 1)


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


class CriticNet(nn.Module):
    def __init__(self, n_in, n_out, hid1_size=128, hid2_size=32):
        super(CriticNet, self).__init__()
        self.layer1 = nn.Linear(n_in, hid1_size)
        self.layer2 = nn.Linear(hid1_size, hid2_size)
        self.v = nn.Linear(hid2_size, n_out)

    def forward(self, x_in):
        hidden = F.relu(self.layer1(x_in))
        hidden = F.relu(self.layer2(hidden))
        return self.v(hidden)


def evaluate(env, agent, n_games=1):
    """ Plays n_games full games, Returns mean reward. """
    t_max = env.spec.timestep_limit or 1000
    rewards = []

    for _ in range(n_games):
        s = env.reset()
        reward = 0.0
        for _ in range(t_max):
            action = agent.get_action(np.array([s]))[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)

    return np.mean(rewards)


def evaluation_thread(env, device, in_queue, sout_queue):
    # each process runs multiple instances of the environment, round-robin
    print("start evaluation process:", os.getpid())
    agent = PGAgent(None, device)

    while True:
        step_idx, model = in_queue.get()
        agent.model = model
        score = evaluate(env, agent, n_games=5)
        print("Step {}: with score : {:.3f}".format(step_idx, score))
        if score >= WIN_SCORE:
            torch.save(model.state_dict(), "checkpoint_win.pt")
            break
        out_queue.put([step_idx, score])


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# for training
STEPS = 20000
LEARNING_RATE = 0.001
BATCH_SIZE = 32
ENV_SIZE = 32
BETA = 0.001
GAMMA = 0.99
TAU = 0.02
GAME = "LunarLanderContinuous-v2"
# GAME = 'BipedalWalker-v2'
WIN_SCORE = 200

batch_envs = BatchEnv(GAME,ENV_SIZE)
# for evaluation
eval_env = gym.make(GAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x, y = eval_env.observation_space.shape[0], eval_env.action_space.shape[0]

actor = ActorNet(x, y).to(device)
actor_eval = ActorNet(x, y).to(device)
critic = CriticNet(x, 1).to(device)
critic_target = CriticNet(x, 1).to(device)
critic_target.load_state_dict(critic.state_dict())
actor_eval.share_memory()
actor_eval.load_state_dict(actor.state_dict())

agent = PGAgent(actor, device)

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)

eval_env.seed(0)
torch.random.manual_seed(0)
batch_envs.reset()

writer = SummaryWriter(comment="a2c_continuous")

eval_inqueue = mp.Queue(maxsize=1)
eval_outqueue = mp.Queue(maxsize=1)


eval_proc = mp.Process(target=evaluation_thread, args=(eval_env, device,eval_inqueue,eval_outqueue))
eval_proc.start()

try:

    for step_idx in range(STEPS):

        actions = agent.get_action(batch_envs.states)
        batch_transitions = batch_envs.step(actions)

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
        log_probs_t = log_probs_t.clamp(min=-20) # for numerical stability
        advantages_t = (target_states_v - predicted_states_v).detach()
        J_actor = (advantages_t.unsqueeze(-1) * log_probs_t).mean()

        # entropy
        entropy = m.entropy().mean()
        L_actor = -J_actor - entropy * BETA

        optimizer_actor.zero_grad()
        L_actor.backward()
        optimizer_actor.step()

        # update actor_env
        if eval_inqueue.empty():
            actor_eval.load_state_dict(actor.state_dict())
            eval_inqueue.put([step_idx, actor_eval])

        # smooth update target
        for target_param, new_param in zip(critic_target.parameters(), critic.parameters()):
            target_param.data.copy_(target_param.data * (1 - TAU) + new_param.data * TAU)

        writer.add_scalar("Entropy", entropy, step_idx)
        writer.add_scalar("Critic_Loss", L_critic, step_idx)
        writer.add_scalar("Actor_Loss", L_actor, step_idx)
        writer.add_scalar("V",predicted_states_v.mean(),step_idx)

        # read evaluation result
        if not eval_outqueue.empty():
            idx, score = eval_outqueue.get()
            writer.add_scalar("Score", score, idx)
            if score >= WIN_SCORE:
                print("Reach the target score 300 of 5 games")
                break


finally:

    eval_proc.terminate()
    eval_proc.join()

