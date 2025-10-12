# Training Data
# Author:   K. E. Brown, Chad GPT.
# First:    2025-10-03
# Updated:  2025-10-08

# training.py
import tensorflow as tf
from network import ActorCriticNetwork


class Trainer:
    def __init__(self, input_size, hidden_size, action_size, lr=0.002, gamma=0.99, entropy_beta=0.01):
        self.network = ActorCriticNetwork(input_size, hidden_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.memory = []
        

    def store(self, obs, action, reward, next_obs, alive):
        self.memory.append((obs, action, reward, next_obs, alive))


    def learn(self, n_steps=3, clip_delta=1.0):
        if len(self.memory) < n_steps:
            return

        obs_list, act_list, rew_list, next_obs_list, alive_list = zip(*self.memory)

        for t in range(len(self.memory) - n_steps):
            G = sum((self.gamma ** k) * rew_list[t + k] for k in range(n_steps))
            done = 0.0 if alive_list[t + n_steps - 1] else 1.0

            obs = tf.convert_to_tensor([obs_list[t]], dtype=tf.float32)
            next_obs = tf.convert_to_tensor([next_obs_list[t + n_steps - 1]], dtype=tf.float32)

            with tf.GradientTape() as tape:
                policy, value = self.network(obs)
                _, next_value = self.network(next_obs)

                target = G + (1 - done) * (self.gamma ** n_steps) * next_value
                advantage = tf.clip_by_value(target - value, -clip_delta, clip_delta)

                action_prob = policy[0, act_list[t]]
                log_prob = tf.math.log(action_prob + 1e-8)
                actor_loss = -log_prob * tf.stop_gradient(advantage)
                critic_loss = tf.square(advantage)
                entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-8))

                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_beta * entropy

            grads = tape.gradient(total_loss, self.network.trainable_variables)
            if None not in grads:
                self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        self.memory = []
