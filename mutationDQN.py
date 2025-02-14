# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# from collections import deque

# # Define the Neural Network for DQN
# class DQN(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, output_size)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# # DQN Agent
# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.95  # Discount factor
#         self.epsilon = 1.0  # Exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001

#         self.model = DQN(state_size, action_size)
#         self.target_model = DQN(state_size, action_size)
#         self.update_target_model()

#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
#         self.loss_fn = nn.MSELoss()

#     def update_target_model(self):
#         self.target_model.load_state_dict(self.model.state_dict())

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)  # Explore
#         state_tensor = torch.FloatTensor(state).unsqueeze(0)
#         q_values = self.model(state_tensor)
#         return torch.argmax(q_values).item()  # Exploit

#     def replay(self, batch_size):
#         if len(self.memory) < batch_size:
#             return

#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             state = torch.FloatTensor(state).unsqueeze(0)
#             next_state = torch.FloatTensor(next_state).unsqueeze(0)
#             target = self.model(state).detach()

#             if done:
#                 target[0][action] = reward
#             else:
#                 next_q_values = self.target_model(next_state).detach()
#                 target[0][action] = reward + self.gamma * torch.max(next_q_values).item()

#             q_values = self.model(state)
#             loss = self.loss_fn(q_values, target)

#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

# # Custom Environment
# class CustomEnvironment:
#     def __init__(self):
#         self.action_space = 2
#         self.state = [1, 4, 5, 6, 8, 2]

#     def reset(self):
#         self.state = [1, 4, 5, 6, 8, 2]
#         return self.state

#     def step(self, action):
#         # Example reward structure with variability
#         base_rewards = [9900, 200]
#         variability = np.random.randint(-100, 100)
#         reward = base_rewards[action] + variability

#         # Simulate next state and episode termination
#         next_state = [x + action for x in self.state]
#         done = random.random() < 0.1  # Random chance of episode ending
#         return next_state, reward, done

# # Training Loop
# if __name__ == "__main__":
#     env = CustomEnvironment()
#     agent = DQNAgent(state_size=len(env.state), action_size=env.action_space)

#     episodes = 1000
#     batch_size = 32

#     for e in range(episodes):
#         state = env.reset()
#         total_reward = 0

#         while True:
#             action = agent.act(state)
#             next_state, reward, done = env.step(action)
#             total_reward += reward

#             agent.remember(state, action, reward, next_state, done)
#             state = next_state

#             if done:
#                 agent.update_target_model()
#                 break

#         agent.replay(batch_size)
#         print(f"Episode {e + 1}: Total Reward = {total_reward}")

#     print("\nTraining complete! Test the model:")
#     test_state = [1, 4, 5, 6, 8, 2]
#     action = agent.act(test_state)
#     print(f"For test state {test_state}, the chosen action is {action}")




import torch
import torch.nn as nn
import torch.optim as optim
import random

class ActionValueNetwork(nn.Module):
    def __init__(self, list_size, action_size):
        super(ActionValueNetwork, self).__init__()
        self.fc1 = nn.Linear(list_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, list1, list2):
        x = torch.cat((list1, list2), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class DQNAgent:
    def __init__(self, list_size, action_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1,
                 learning_rate=0.001, memory_size=10000, batch_size=32):
        self.list_size = list_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.model = ActionValueNetwork(list_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.replay_memory = []

    def reward_calculation(self, action, list1, list2):
        if action == 0:  # SumOfList
            reward = -abs(sum(list1) - sum(list2))
        elif action == 1:  # MultiplicationOfList
            reward = abs(sum([a * b for a, b in zip(list1, list2)]))
        return reward

    def choose_action(self, state):
        list1, list2 = state
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.model(list1, list2)
                return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state):
        self.replay_memory.append((state, action, reward, next_state))
        if len(self.replay_memory) > self.memory_size:
            self.replay_memory.pop(0)

    def train(self):
        if len(self.replay_memory) < self.batch_size:
            return

        batch = random.sample(self.replay_memory, self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states = zip(*batch)

        batch_list1 = torch.cat([s[0] for s in batch_states])
        batch_list2 = torch.cat([s[1] for s in batch_states])
        batch_next_list1 = torch.cat([s[0] for s in batch_next_states])
        batch_next_list2 = torch.cat([s[1] for s in batch_next_states])
        batch_actions = torch.tensor(batch_actions)
        batch_rewards = torch.tensor(batch_rewards).float()

        with torch.no_grad():
            target_q_values = batch_rewards + self.gamma * torch.max(
                self.model(batch_next_list1, batch_next_list2), dim=1
            )[0]

        current_q_values = self.model(batch_list1, batch_list2)
        current_q_values = current_q_values.gather(1, batch_actions.unsqueeze(1)).squeeze()

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def evaluate(self, list1, list2):
        with torch.no_grad():
            list1_tensor = torch.tensor([list1], dtype=torch.float32)
            list2_tensor = torch.tensor([list2], dtype=torch.float32)
            q_values = self.model(list1_tensor, list2_tensor)
            best_action = torch.argmax(q_values).item()
            return best_action, q_values.numpy()

    def train_model(self, num_episodes=1):
        for episode in range(num_episodes):
            list1 = torch.randint(1, 10, (1, self.list_size)).float()
            list2 = torch.randint(1, 10, (1, self.list_size)).float()
            print(list2)
            print(list2)
            state = (list1, list2)

            for t in range(10):
                action = self.choose_action(state)
                reward = self.reward_calculation(action, list1[0].tolist(), list2[0].tolist())
                next_state = state

                self.store_experience(state, action, reward, next_state)
                self.train()

            self.decay_epsilon()

            if episode % 100 == 0:
                print(f"Episode {episode}, Epsilon: {self.epsilon:.2f}")

        print("Training completed.")

# Example usage
list_size = 5
action_size = 2
agent = DQNAgent(list_size, action_size)

# Train the agent
agent.train_model(num_episodes=90)

# Evaluate the trained model
list1 = [1, 2, 3, 4, 5]
list2 = [5, 6, 4, 6, 2]
best_action, q_values = agent.evaluate(list1, list2)
actions = ["SumOfList", "MultiplicationOfList"]
print(f"Best Action: {actions[best_action]}")
print(f"Q-values: {q_values}")
