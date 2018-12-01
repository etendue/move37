from torch.distributions.categorical import Categorical
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from tensorboardX import SummaryWriter

import torch.multiprocessing as mp
import math
import os

def run_step(env,a):
    next_s,r,done, _ = env.step(a)
    if done:
        next_s = env.reset()
    return Transition(None,a,r,next_s,done)

class PGAgent():
    def __init__(self,model,device):
        """A simple PG agent"""
        self.model = model
        self.device = device

    def get_action(self, state):
        """interface for Agent"""
        s = torch.FloatTensor(state).to(self.device)
        logits = self.model(s).detach()
        m = Categorical(logits = logits)
        return m.sample().cpu().data.numpy().tolist()[0]

def evaluate(env, agent, n_games=1):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    t_max = env.spec.timestep_limit or 1000
    rewards = []

    for _ in range(n_games):
        s = env.reset()
        reward = 0.0
        for _ in range(t_max):
            action = agent.get_action(np.array([s]))
            s, r, done, _ = env.step(action)
            reward += r
            if done: break

        rewards.append(reward)

    return np.mean(rewards)

def data_func(net,device, train_queue):
    # each process runs multiple instances of the environment, round-robin
    print("start work process:",os.getpid())
    envs = [gym.make(GAME) for _ in range(ENVS_PER_PROCESS)]
    agent = PGAgent(net,device)

    states= [env.reset() for env in envs]
    while True:
        for i,env in enumerate(envs):
            s0 = states[i]
            a0 = agent.get_action([s0])
            a = a0
            r_total = 0.0
            for j in range(N_STEP):
                next_s,r,done, _ = env.step(a)
                r_total = r_total + r * GAMMA**j
                if done:
                    next_s = env.reset()
                    break
                a = agent.get_action([next_s])

            states[i] = next_s
            train_queue.put(Transition(s0,a0,r_total,next_s, done))

Transition = namedtuple('Transition',('state', 'action', 'reward','next_state','done'))

# for training
STEPS = 20000
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
BETA = 0.1
GAMMA = 1.0
TAU = 0.05
N_STEP = 1
GAME = "LunarLander-v2"
#GAME = "CartPole-v0"

TOTAL_ENVS = 32
PROCESSES_COUNT = max((mp.cpu_count() - 2),1)
ENVS_PER_PROCESS = math.ceil(TOTAL_ENVS / PROCESSES_COUNT)


batch_envs = [gym.make(GAME) for _ in range(BATCH_SIZE)]
# for evaluation
eval_env = gym.make(GAME)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
x, y = eval_env.observation_space.shape[0], eval_env.action_space.n

actor = torch.nn.Sequential(nn.Linear(x,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,y)).to(device)
critic = torch.nn.Sequential(nn.Linear(x,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,1)).to(device)
critic_target = torch.nn.Sequential(nn.Linear(x,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,1)).to(device)

actor.share_memory()
agent = PGAgent(actor,device)

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE*5)

eval_env.seed(0)
torch.random.manual_seed(0)

train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
data_proc_list = []
# Spawn processes to run data_func
for _ in range(PROCESSES_COUNT):
    data_proc = mp.Process(target=data_func, args=(actor, device,train_queue))
    data_proc.start()
    data_proc_list.append(data_proc)

batch = []


writer = SummaryWriter(comment="a3c")


try:
    for step_idx in range(STEPS):

        while len(batch) < BATCH_SIZE:
            train_entry = train_queue.get()
            batch.append(train_entry)

        transitions = Transition(*zip(*batch))
        batch.clear()
        states_t = torch.FloatTensor(transitions.state).to(device)
        actions_t = torch.LongTensor(transitions.action).to(device)
        rewards_t = torch.FloatTensor(transitions.reward).to(device)
        next_states_t = torch.FloatTensor(transitions.next_state).to(device)
        done_t = torch.FloatTensor(transitions.done).to(device)

        logits_t = actor(states_t)
        m = Categorical(logits=logits_t)

        # critic loss
        predicted_states_v = critic(states_t).squeeze()
        with torch.no_grad():
            predicted_next_states_v = critic_target(next_states_t).squeeze() * (1 - done_t)
            target_states_v = predicted_next_states_v * GAMMA + rewards_t

        L_critic = F.smooth_l1_loss(predicted_states_v, target_states_v)

        optimizer_critic.zero_grad()
        L_critic.backward()
        optimizer_critic.step()


        # actor loss

        log_probs_t = m.log_prob(actions_t)
        advantages_t = (target_states_v - predicted_states_v).detach()
        J_actor = (advantages_t * log_probs_t).mean()

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
            #critic_target.load_state_dict(critic.state_dict())
            score = evaluate(eval_env, agent, n_games=5)
            print("Step {}: with score : {:.3f}".format(step_idx, score))
            writer.add_scalar("Score", score, step_idx)

            if score>= 200:
                print("Reach the target score 200 of 5 games")
                break

finally:
    for p in data_proc_list:
        p.terminate()
        p.join()


