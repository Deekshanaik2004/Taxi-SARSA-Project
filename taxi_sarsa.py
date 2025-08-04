import gymnasium as gym
import numpy as np
import random
import time
import os

# Create Taxi-v3 environment with GUI
env = gym.make("Taxi-v3", render_mode="human")

# Q-table path
q_table_file = "sarsa_q_table.npy"

# Initialize empty Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 10000
max_steps = 100

# ✅ Load existing Q-table if available
if os.path.exists(q_table_file):
    print("📂 Found saved Q-table. Loading it...")
    q_table = np.load(q_table_file)
else:
    print("🚀 No saved model found. Training using SARSA...")

    # SARSA training
    for episode in range(episodes):
        state, _ = env.reset()
        action = env.action_space.sample() if random.uniform(0, 1) < epsilon else np.argmax(q_table[state])

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_action = env.action_space.sample() if random.uniform(0, 1) < epsilon else np.argmax(q_table[next_state])

            q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])

            state = next_state
            action = next_action

            if done:
                break

    # ✅ Save Q-table after training
    np.save(q_table_file, q_table)
    print("💾 Q-table saved to:", q_table_file)

# ✅ GUI Test
def test_agent_gui(episodes=5):
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        print(f"\n🎮 Episode {ep + 1}")
        time.sleep(1)

        while not done:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            time.sleep(0.5)

    env.close()

# 👇 Run GUI test
test_agent_gui()
