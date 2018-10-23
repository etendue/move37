import numpy as np
import gym
from gym import wrappers
import os


class Hp():
    # Hyperparameters
    def __init__(self,
                 epsiode_num = 1000,
                 episode_length=2000,
                 learning_rate=0.02,
                 num_deltas=16,
                 num_best_deltas=16,
                 noise=0.03,
                 seed=1,
                 env_name='BipedalWalker-v2',
                 record_every=50):
        self.epsiode_num = epsiode_num
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.num_deltas = num_deltas
        self.num_best_deltas = num_best_deltas
        assert self.num_best_deltas <= self.num_deltas
        self.noise = noise
        self.seed = seed
        self.env_name = env_name
        self.record_every = record_every


class SampleNormalizer():
    # Normalizes the inputs
    def __init__(self, nb_inputs):
        self.n = 0
        self.mu = np.zeros(nb_inputs)
        self.S = np.zeros(nb_inputs)

    def update(self, x):
        self.n += 1.0
        mu_ = self.mu + (x - self.mu) / self.n
        self.S += (x - self.mu) * (x - mu_)
        self.mu = mu_

        return self.normalized_x(x)

    @property
    def mean(self):
        return self.mu

    @property
    def var(self):
        return self.S / self.n if self.n > 1 else 0

    @property
    def std(self):
        return np.sqrt(self.var)

    def normalized_x(self, x):
        return (x - self.mu) / (self.std.clip(min=1e-2))


class PerceptronModel():
    def __init__(self, x_size, action_size, learning_rate = 0.1):
        self.W = np.zeros((action_size, x_size))
        self.learning_rate = learning_rate

    def predict(self,x, noise = None):
        return (self.W+noise).dot(x) if noise is not None else self.W.dot(x)

    def update(self, rollouts, reward_std):
        # sigma_rewards is the standard deviation of the rewards
        r_pos, r_neg, deltas = rollouts
        step = np.average(deltas, axis=0, weights=(r_pos-r_neg))
        self.W += self.learning_rate * step/reward_std


class ArsTrainer():
    def __init__(self,
                 hp=None,
                 input_size=None,
                 output_size=None,
                 normalizer=None,
                 model=None,
                 monitor_dir=None):

        self.hp = hp or Hp()
        np.random.seed(self.hp.seed)
        self.env = gym.make(self.hp.env_name)
        if monitor_dir is not None:
            should_record = lambda i: self.record_video
            self.env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)
        self.hp.episode_length = self.env.spec.timestep_limit or self.hp.episode_length
        self.input_size = input_size or self.env.observation_space.shape[0]
        self.output_size = output_size or self.env.action_space.shape[0]
        self.normalizer = normalizer or SampleNormalizer(self.input_size)
        self.model = model or PerceptronModel(self.input_size, self.output_size)
        self.record_video = False

    # Evaluate the model
    def evaluate(self):
        state = self.env.reset()
        done = False
        num_plays = 0.0
        sum_rewards = 0.0
        while not done and num_plays < self.hp.episode_length:
            normalized_state = self.normalizer.normalized_x(state)
            action = self.model.predict(state)
            state, reward, done, _ = self.env.step(action)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
        return sum_rewards

    # Explore the model
    def explore(self, delta=None):

        # explore positive direction
        state = self.env.reset()
        done = False
        num_plays = 0
        sum_rewards_pos = 0.0
        while not done and num_plays < self.hp.episode_length:
            normalized_state = self.normalizer.update(state)
            action = self.model.predict(normalized_state, noise=delta * self.hp.noise)
            state, reward, done, _ = self.env.step(action)
            sum_rewards_pos += np.clip(reward, a_min=-1, a_max=1)
            num_plays += 1

        # explore negative direction
        state = self.env.reset()
        done = False
        num_plays = 0.0
        sum_rewards_neg = 0.0
        while not done and num_plays < self.hp.episode_length:
            normalized_state = self.normalizer.update(state)
            action = self.model.predict(normalized_state, noise=-delta * self.hp.noise)
            state, reward, done, _ = self.env.step(action)
            sum_rewards_neg += np.clip(reward, a_min=-1, a_max=1)
            num_plays += 1

        return sum_rewards_pos, sum_rewards_neg

    def train(self):
        for eposide in range(self.hp.epsiode_num):
            # initialize the random noise deltas and the positive/negative rewards
            deltas = np.random.randn(self.hp.num_deltas, *self.model.W.shape)
            positive_rewards = np.zeros(self.hp.num_deltas, dtype=np.float)
            negative_rewards = np.zeros(self.hp.num_deltas, dtype=np.float)

            # play an episode each with positive deltas and negative deltas, collect rewards
            # TODO: parallelize the exploration
            for i in range(self.hp.num_deltas):
                positive_rewards[i], negative_rewards[i] = self.explore(delta=deltas[i])

            # Compute the standard deviation of all rewards
            reward_std = np.hstack([positive_rewards, negative_rewards]).std()

            # Sort the rollouts by the max(r_pos, r_neg) and select the deltas with best rewards
            reward_maximum = np.maximum(positive_rewards, negative_rewards)
            idx = np.argsort(reward_maximum)
            n = self.hp.num_best_deltas
            rollouts = positive_rewards[idx[-n:]], negative_rewards[idx[-n:]], deltas[idx[-n:]]
            self.model.update(rollouts, reward_std)

            # Only record video during evaluation, every n steps
            if eposide % self.hp.record_every == 0:
                self.record_video = True
            # Play an episode with the new weights and print the score
            reward_evaluation = self.evaluate()
            print('Eposiode: ', eposide, 'Reward: ', reward_evaluation)
            self.record_video = False


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# Main code
ENV_NAME = 'BipedalWalker-v2'
if __name__ == '__main__':
    videos_dir = mkdir('.', 'videos')
    monitor_dir = mkdir(videos_dir, ENV_NAME)
    hp = Hp(seed=1980, env_name=ENV_NAME)
    trainer = ArsTrainer(hp=hp, monitor_dir=monitor_dir)
    trainer.train()