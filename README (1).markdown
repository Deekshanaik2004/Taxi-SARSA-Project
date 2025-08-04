# SARSA Taxi-v3 Reinforcement Learning

This project implements a SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm to train an agent in the `Taxi-v3` environment from OpenAI Gymnasium. The agent learns to navigate a taxi to pick up and drop off passengers efficiently, with a graphical user interface (GUI) for visualization.

## Features
- **Environment**: Uses `Taxi-v3` from Gymnasium with `render_mode="human"` for GUI visualization.
- **Algorithm**: SARSA, an on-policy reinforcement learning method.
- **Q-table**: Persistently stored and loaded using NumPy (saved as `sarsa_q_table.npy`).
- **Training**: Trains for 10,000 episodes with a maximum of 100 steps per episode if no pre-trained Q-table exists.
- **Testing**: Includes a GUI test mode to visualize the agent's performance for 5 episodes.
- **Hyperparameters**:
  - Learning rate (`alpha`): 0.1
  - Discount factor (`gamma`): 0.9
  - Exploration rate (`epsilon`): 0.1

## Requirements
- Python 3.x
- Gymnasium (`pip install gymnasium`)
- NumPy (`pip install numpy`)

## Usage
1. **Install dependencies**:
   ```bash
   pip install gymnasium numpy
   ```
2. **Run the script**:
   - Save the provided Python code as `sarsa_taxi.py`.
   - Execute the script:
     ```bash
     python sarsa_taxi.py
     ```
   - If `sarsa_q_table.npy` exists, the script loads the pre-trained Q-table and runs the GUI test.
   - If no Q-table is found, the script trains the agent using SARSA for 10,000 episodes, saves the Q-table, and then runs the GUI test.
3. **GUI Test**:
   - The script runs 5 test episodes with a 0.5-second delay between actions for visualization.
   - The Gymnasium window displays the taxi's movements.

## Files
- `sarsa_taxi.py`: Main script containing the SARSA algorithm, training, and GUI testing logic.
- `sarsa_q_table.npy`: Saved Q-table file (generated after training).

## How It Works
- **Training**:
  - If no Q-table exists, the agent trains using SARSA, updating the Q-table based on the current state, action, reward, next state, and next action.
  - The agent uses an epsilon-greedy policy for exploration vs. exploitation.
  - The Q-table is saved after training to avoid retraining.
- **Testing**:
  - The pre-trained Q-table is used to select the best actions.
  - The GUI displays the taxi navigating the environment to pick up and drop off passengers.
- **Persistence**:
  - The Q-table is saved as `sarsa_q_table.npy` and loaded automatically if available.

## Notes
- The `Taxi-v3` environment is a grid-based world where the taxi must pick up a passenger and drop them off at the correct destination.
- The GUI test mode (`test_agent_gui`) runs 5 episodes to demonstrate the agent's learned policy.
- The script uses a 0.5-second delay in the GUI test for better visualization; adjust `time.sleep(0.5)` if needed.
- To retrain the agent, delete `sarsa_q_table.npy` before running the script.

## Example Output
- If training:
  ```
  🚀 No saved model found. Training using SARSA...
  💾 Q-table saved to: sarsa_q_table.npy
  🎮 Episode 1
  ...
  ```
- If loading:
  ```
  📂 Found saved Q-table. Loading it...
  🎮 Episode 1
  ...
  ```

## Limitations
- The script assumes a local environment with a display for GUI rendering. For headless environments, change `render_mode` to `"ansi"` or similar.
- Training may take time due to the 10,000 episodes. Adjust `episodes` or `max_steps` for faster training.
- The GUI test is designed for visual inspection and may not work in non-GUI environments (e.g., some cloud platforms).

## License
This project is for educational purposes and uses open-source libraries under their respective licenses (Gymnasium, NumPy).