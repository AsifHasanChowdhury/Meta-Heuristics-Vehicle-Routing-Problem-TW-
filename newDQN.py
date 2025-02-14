import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define the Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()  # Exploit

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            target = self.model(state).detach()

            if done:
                target[0][action] = reward
            else:
                next_q_values = self.target_model(next_state).detach()
                target[0][action] = reward + self.gamma * torch.max(next_q_values).item()

            q_values = self.model(state)
            loss = self.loss_fn(q_values, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Custom Environment
class CustomEnvironment:
    def __init__(self):
        self.action_space = 2  # AddDetails (0) and SubtractDetails (1)

    def reset(self, list1, list2):
        self.list1 = list1
        self.list2 = list2
        return (self.list1, self.list2)

    def step(self, action):
        if action == 0:  # AddDetails
            result = [x + y for x, y in zip(self.list1, self.list2)]
            reward = sum(result)  # Example: reward is the sum of the results
        elif action == 1:  # SubtractDetails
            result = [x - y for x, y in zip(self.list1, self.list2)]
            reward = sum(1 for r in result if r > 0)  # Example: count of positives

        next_state = (self.list1, self.list2)  # State remains the same for simplicity
        done = True  # One-step episode
        return next_state, reward, done

# Training Loop
def train_dqn():
    env = CustomEnvironment()
    state_size = 6  # Length of list1 + list2 (3 + 3 for this example)
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)

    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        list1 = [random.randint(1, 10) for _ in range(3)]
        list2 = [random.randint(1, 10) for _ in range(3)]

        state = env.reset(list1, list2)
        state_flattened = np.array(state[0] + state[1])  # Flatten state for input to the model
        total_reward = 0

        while True:
            action = agent.act(state_flattened)
            next_state, reward, done = env.step(action)
            next_state_flattened = np.array(next_state[0] + next_state[1])

            agent.remember(state_flattened, action, reward, next_state_flattened, done)
            state_flattened = next_state_flattened
            total_reward += reward

            if done:
                agent.update_target_model()
                break

        agent.replay(batch_size)
        print(f"Episode {e + 1}: Total Reward = {total_reward}")

    print("\nTraining complete! Test the model:")
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    state = env.reset(list1, list2)
    state_flattened = np.array(state[0] + state[1])
    action = agent.act(state_flattened)
    print(f"For test state (list1={list1}, list2={list2}), the chosen action is {action}")

# Run the training
train_dqn()
