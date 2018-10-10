import numpy as np
import gym
from tqdm import tqdm
from collections import  deque

class MonteCarloAgent:
    def __init__(self, state_space, action_space, gamma= 1.0, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = 0.7
        self.V = np.zeros(state_space)
        self.pi = np.ones(state_space, dtype=np.int) * -1
        self.states_actions = np.zeros((state_space, action_space))
        #self.states_actions_history =  np.zeros((state_space, action_space))

    def StateValueIteration(self, sar_rollout):
        G_t_ = 0
        visited_state_action = set()
        visited_state = set()

        for s, a, r in reversed(sar_rollout):
            if (s, a) not in visited_state_action:
                visited_state_action.add((s, a))
                #self.state_actions_history[s][a] = r + G_t_ * self.gamma
                #self.states_actions[s][a] = np.mean(np.array(self.state_actions_history[s][a]))
                self.states_actions[s][a] = self.states_actions[s][a] * self.alpha + (r+G_t_ * self.gamma) *(1 - self.alpha)
                G_t_ = np.max(self.states_actions[s])

                visited_state.add(s)

        # update policy for updated state
        for s in visited_state:
            self.pi[s] = np.argmax(self.states_actions[s])
            self.V[s] = np.max(self.states_actions[s])

    def GetPolicyVisual(self):

        literal = []
        for a in self.pi:
            if a == 0:
                literal.append('\u2191')
            elif a == 1:
                literal.append('\u2192')
            elif a == 2:
                literal.append('\u2193')
            elif a == 3:
                literal.append('\u2190')
            else:
                literal.append('O')

        return literal


def cliff_walking_rollout(env, policy, epsilon=0.1, max_step=10000):
    start = env.reset()
    s = start
    sar = []

    for _ in range(max_step):

        a = policy[s]
        sample = np.random.uniform()
        if sample < epsilon or a == -1:
            a = env.action_space.sample()

        s_next, r, done, _ = env.step(a)
        sar.append((s, a, r))
        s = s_next
        if done:
            #print("Goal")
            break

    return sar


env = gym.make('CliffWalking-v0')
start = env.reset()

agent= MonteCarloAgent(env.observation_space.n,env.action_space.n,gamma=0.99)
np.set_printoptions(precision=1)

agent.epsilon = 0.5
for i in tqdm(range(50000)):
    rollout = cliff_walking_rollout(env,agent.pi,agent.epsilon)
    agent.StateValueIteration(rollout)

    if agent.epsilon > 0.01 and i % 1000 == 999:
        agent.epsilon = agent.epsilon * 0.5

    if i % 100 == 99:

        print("_______")
        print(np.array(agent.GetPolicyVisual()).reshape((4,12)))
        print(agent.V.reshape((4,12)))

#print(agent.states_actions)
