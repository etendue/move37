# Forked from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On

import math
import gym
import numpy as np
import collections
import os
import copy
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions.categorical import Categorical

GAMMA = 0.99
LEARNING_RATE = 0.0005
ENTROPY_BETA = 0.01
BATCH_SIZE = 64
TAU = 0.05
N_STEP = 8

TOTAL_ENVS = 48
PROCESSES_COUNT = max((mp.cpu_count() - 2),1)
ENVS_PER_PROCESS = math.ceil(TOTAL_ENVS / PROCESSES_COUNT)


ENV_NAME = "LunarLander-v2"
REWARD_BOUND = 200


class PGAgent():
    def __init__(self,model,device):
        """A simple PG agent"""
        self.model = model
        self.device = device

    def get_action(self, state):
        """interface for Agent"""
        s = torch.from_numpy(np.array([state])).to(self.device)
        with torch.no_grad():
            logits = self.model(s)
            m = Categorical(logits = logits)
            return m.sample().cpu().data.numpy()[0]


Transition = collections.namedtuple('Transition',('state', 'action', 'reward','next_state','done'))


def evaluate(env, agent, n_games=1):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    t_max = env.spec.timestep_limit or 10000
    rewards = []

    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            action = agent.get_action(s)
            s, r, done, _ = env.step(action)

            reward += r
            if done: break

        rewards.append(reward)

    return np.mean(rewards)

# data_func is forked and run by each process
# It grabs one experience transition from an environment and adds it to the training queue
def data_func(net,device, train_queue):
    # each process runs multiple instances of the environment, round-robin
    print("start work process:",os.getpid())
    envs = [gym.make(ENV_NAME) for _ in range(ENVS_PER_PROCESS)]
    agent = PGAgent(net,device)

    states= [env.reset() for env in envs]
    while True:

        for i,env in enumerate(envs):
            s0 = states[i]
            a0 = agent.get_action(s0)
            a = a0
            r_total = 0.0
            for j in range(N_STEP):
                next_s,r,done, _ = env.step(a)
                r_total = r_total + r * GAMMA**j
                if done:
                    next_s = env.reset()
                    break
                a = agent.get_action(next_s)

            states[i] = next_s
            train_queue.put(Transition(s0,a0,r_total,next_s, done))

if __name__ == "__main__":

    mp.set_start_method('spawn')

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device ="cpu"

    env = gym.make(ENV_NAME)
    actor = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256,env.action_space.n)
    ).to(device)

    critic = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 64),
        nn.ReLU(),
        nn.Linear(64,128),
        nn.ReLU(),
        nn.Linear(128,1)
    ).to(device)

    critic_target = copy.deepcopy(critic)

    actor.share_memory()
    agent = PGAgent(actor,device)

    print(actor)
    print(critic)

    optim_actor = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    optim_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    # Spawn processes to run data_func
    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=data_func, args=(actor, device,train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    step_idx = 0

    writer = SummaryWriter(comment="NStep_{}.log".format(N_STEP))

    # envs = [gym.make(ENV_NAME) for _ in range(ENVS_PER_PROCESS)]
    # agent = PGAgent(actor, device)
    # states = [env.reset() for env in envs]

    try:
        while True:

            # for i, env in enumerate(envs):
            #     s = states[i]
            #     a = agent.get_action(s)
            #     next_s, r, done, _ = env.step(a)
            #     if done:
            #         next_s = env.reset()
            #     states[i] = next_s
            #     batch.append(Transition(s, a, r, next_s, done))
            # Get one transition from the training queue
            train_entry = train_queue.get()

            # keep receiving data until one batch is full
            batch.append(train_entry)
            step_idx += 1

            if len(batch) < BATCH_SIZE:
                continue


            transitions = Transition(*zip(*batch))
            states_t = torch.FloatTensor(transitions.state).to(device)
            actions_t = torch.LongTensor(transitions.action).to(device)
            rewards_t = torch.FloatTensor(transitions.reward).to(device)
            next_states_t = torch.FloatTensor(transitions.next_state).to(device)
            done_t = torch.FloatTensor(transitions.done).to(device)

            batch.clear()

            # critic loss
            predicted_states_v = critic(states_t).squeeze()
            with torch.no_grad():
                next_states_v = critic_target(next_states_t).squeeze() * (1 - done_t)
                target_states_v = next_states_v * GAMMA + rewards_t

            L_critic = F.smooth_l1_loss(predicted_states_v, target_states_v)

            optim_critic.zero_grad()
            L_critic.backward()
            optim_critic.step()
            # smooth update target
            for target_param, new_param in zip(critic_target.parameters(),critic.parameters()):
                target_param.data = target_param.data * (1- TAU) + new_param.data*TAU


            # actor loss
            logits = actor(states_t)
            m = Categorical(logits =logits)
            log_probs_t = m.log_prob(actions_t)
            advantages_t = (target_states_v - predicted_states_v).detach()
            J_actor = (advantages_t * log_probs_t).mean()

            # entropy
            entropy = m.entropy().mean()

            L_actor = -J_actor - entropy * ENTROPY_BETA
            optim_actor.zero_grad()
            L_actor.backward()
            optim_actor.step()

            writer.add_scalar("Entropy", entropy,step_idx)
            writer.add_scalar("Critic_Loss",L_critic,step_idx)
            writer.add_scalar("Actor_Loss",L_actor,step_idx)
            writer.add_scalar("V_mean",predicted_states_v.mean(),step_idx)

            if step_idx % 512 == 0:
                score = evaluate(env, agent, n_games=5)
                print("Step {}: with score : {:.3f}".format(step_idx,score))
                writer.add_scalar("Score",score,step_idx)
                if score >= REWARD_BOUND:
                    score = evaluate(env, agent, n_games=5)
                    if score >= REWARD_BOUND:
                        break

    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()
