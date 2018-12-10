from torch.distributions.normal import Normal
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from tensorboardX import SummaryWriter
from collections import deque
import torch.multiprocessing as mp
import os


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
        return m.sample().cpu().data.numpy()


class ActorNet(nn.Module):
    def __init__(self, n_in, n_out, hid1_size=128, hid2_size=128):
        super(ActorNet, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
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


def work_thread(agent, event, t_queue, score_queue):
    # each process runs multiple instances of the environment, round-robin
    print("start work process:", os.getpid())
    env = gym.make(GAME)
    t_max = env.spec.timestep_limit or 1000

    while True:
        total_reward = 0.0
        s = env.reset()
        for i in range(t_max):
            event.wait()
            a = agent.get_action([s])[0]
            next_s, r, done, _ = env.step(a)
            total_reward += r
            # special case that time/step limit is reached.
            if i == t_max - 1:
                score_queue.put([total_reward, r, i])
                done = False
                t_queue.put(Transition(s, a, r, next_s, done))
                break

            if done:
                score_queue.put([total_reward, r, i])
                t_queue.put(Transition(s, a, r, next_s, done))
                break

            t_queue.put(Transition(s, a, r, next_s, done))
            s = next_s


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# for training
STEPS = 200000
LEARNING_RATE = 0.001
BATCH_SIZE = 64

BETA = 0.15
GAMMA = 0.99
TAU = 0.005
GAME = "LunarLanderContinuous-v2"
# GAME = 'BipedalWalker-v2'
WIN_SCORE = 200
CPU_NUM = mp.cpu_count()
WORKER_NUM = max(1, CPU_NUM - 2)

# for evaluation
eval_env = gym.make(GAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x, y = eval_env.observation_space.shape[0], eval_env.action_space.shape[0]

actor = ActorNet(x, y).to(device)
actor.share_memory()

critic = CriticNet(x, 1).to(device)
critic_target = CriticNet(x, 1).to(device)
critic_target.load_state_dict(critic.state_dict())


agent = PGAgent(actor, device)

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)

torch.random.manual_seed(0)


writer = SummaryWriter(comment="a2c_continuous")

t_queue = mp.Queue(maxsize=BATCH_SIZE)
score_queue = mp.Queue(maxsize=WORKER_NUM)
working_event = mp.Event()


proc_list = []
for _ in range(WORKER_NUM):
    work_proc = mp.Process(target=work_thread, args=(agent, working_event, t_queue, score_queue))
    proc_list.append(work_proc)
    work_proc.start()

working_event.set()

batch_transitions = []
score_history = deque(maxlen=5)
bingo = 0

try:

    for step_idx in range(STEPS):

        # collecting transitions
        transition = t_queue.get()
        batch_transitions.append(transition)
        if len(batch_transitions) < BATCH_SIZE:
            continue
        # Pause worker for collecting experience
        working_event.clear()

        # convert transitions to tensor
        batch = Transition(*zip(*batch_transitions))
        states_t = torch.FloatTensor(batch.state).to(device)
        actions_t = torch.FloatTensor(batch.action).to(device)
        next_states_t = torch.FloatTensor(batch.next_state).to(device)
        rewards_t = torch.FloatTensor(batch.reward).to(device)
        done_t = torch.FloatTensor(batch.done).to(device)

        batch_transitions.clear()

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
        advantages_t = (target_states_v - predicted_states_v).detach()
        J_actor = (advantages_t.unsqueeze(-1) * log_probs_t).mean()

        # entropy
        entropy = m.entropy().mean()
        L_actor = -J_actor - entropy * BETA

        optimizer_actor.zero_grad()
        L_actor.backward()
        optimizer_actor.step()

        # removing invalid experience
        while not t_queue.empty():
            t_queue.get()

        # start worker again
        working_event.set()

        # smooth update target
        for target_param, new_param in zip(critic_target.parameters(), critic.parameters()):
            target_param.data.copy_(target_param.data * (1 - TAU) + new_param.data * TAU)

        writer.add_scalar("Entropy", entropy, step_idx)
        writer.add_scalar("Critic_Loss", L_critic, step_idx)
        writer.add_scalar("Actor_Loss", L_actor, step_idx)
        writer.add_scalar("V", predicted_states_v.mean(),step_idx)

        # read evaluation result
        if not score_queue.empty():
            score, last_r, count = score_queue.get()
            score_history.append(score)
            writer.add_scalar("Score", score, step_idx)
            if last_r == 100:
                bingo += 1
            print("Batch {} : Score {:.2f}, steps {}, last r {:.2f}, bingo {}". format(step_idx, score, count,last_r,bingo))

            if np.mean(score_history) >= WIN_SCORE:

                validate_score = evaluate(eval_env, agent, n_games=5)
                if validate_score > WIN_SCORE:
                    print("Reach the target score of 5 games")
                    break


finally:
    for p in proc_list:
        p.terminate()
        p.join()

