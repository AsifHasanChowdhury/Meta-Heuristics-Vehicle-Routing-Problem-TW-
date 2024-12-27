import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import CrossOver as co

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
    def __init__(self,ActionSize,population):
        self.action_space = ActionSize
        self.state = population

    def reset(self,population):
        self.state = population
        return self.state

    def step(self, action,fitnessScoreList):
        # Example reward structure with variability
        base_rewards = fitnessScoreList
        variability = np.random.randint(-100, 100)
        reward = base_rewards[action] + variability

        # Simulate next state and episode termination
        next_state = [x + action for x in self.state]
        done = random.random() < 0.1  # Random chance of episode ending
        return next_state, reward, done

# Training Loop
def ReinforcementDriverMethod(singlePopulation,action_space,fitnessScoreList):
    env = CustomEnvironment(action_space,singlePopulation)
    agent = DQNAgent(state_size=len(env.state), action_size=env.action_space)

    episodes = 5
    batch_size = 32

    for e in range(episodes):
        state = env.reset(singlePopulation)
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action,fitnessScoreList)
            total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                agent.update_target_model()
                break

        agent.replay(batch_size)
        print(f"Episode {e + 1}: Total Reward = {total_reward}")

    print("\nTraining complete! Test the model:")
    test_state = singlePopulation
    action = agent.act(test_state)
    print(f"For test state {test_state}, the chosen action is {action}")

    return action,total_reward


#ReinforcementDriverMethod([1, 4, 5, 6, 8, 2],2,[0.555,0.99999])