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

        # Separate experience replay for each group
        self.replay_memory_group1 = []
        self.replay_memory_group2 = []

        # Define action groups
        self.actions = ["SumOfList", "MultiplicationOfList", "SubtractOfList", "DivisionOfList"]
        self.action_groups = {
            1: [0, 1],  # Group 1: SumOfList, MultiplicationOfList
            2: [2, 3],  # Group 2: SubtractOfList, DivisionOfList
        }

    def choose_action(self, state, action_group=None):
    
        list1, list2 = state

        # Determine available actions based on the group
        available_actions = (
            self.action_groups.get(action_group, list(range(self.action_size)))
        )

        if random.random() < self.epsilon:
            # Choose a random action from the available actions
            return random.choice(available_actions)
        else:
            # Choose the action with the highest Q-value among the available actions
            with torch.no_grad():
                q_values = self.model(list1, list2)
                q_values_filtered = torch.tensor([q_values[0][a] for a in available_actions])
                return available_actions[torch.argmax(q_values_filtered).item()]

    def reward_calculation(self, action, list1, list2):
        if action == 0:  # SumOfList
            reward = -abs(sum(list1) - sum(list2))
        elif action == 1:  # MultiplicationOfList
            reward = abs(sum([a * b for a, b in zip(list1, list2)]))
        elif action == 2:  # SubtractOfList
            reward = -abs(sum(list1) - sum(list2))
        elif action == 3:  # DivisionOfList
            reward = -sum([a / b if b != 0 else 0 for a, b in zip(list1, list2)])
        return reward

    def store_experience(self, state, action, reward, next_state, action_group):
        if action_group == 1:
            replay_memory = self.replay_memory_group1
        elif action_group == 2:
            replay_memory = self.replay_memory_group2
        else:
            return

        replay_memory.append((state, action, reward, next_state))
        if len(replay_memory) > self.memory_size:
            replay_memory.pop(0)

    def train(self, action_group):
        replay_memory = (
            self.replay_memory_group1 if action_group == 1 else
            self.replay_memory_group2 if action_group == 2 else
            None
        )
        if not replay_memory or len(replay_memory) < self.batch_size:
            return

        batch = random.sample(replay_memory, self.batch_size)
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

    def evaluate(self, list1, list2, action_group=None):
        with torch.no_grad():
            list1_tensor = torch.tensor([list1], dtype=torch.float32)
            list2_tensor = torch.tensor([list2], dtype=torch.float32)
            q_values = self.model(list1_tensor, list2_tensor)
            
            available_actions = self.action_groups.get(action_group, list(range(self.action_size)))
            q_values_filtered = torch.tensor([q_values[0][a] for a in available_actions])
            best_action = available_actions[torch.argmax(q_values_filtered).item()]
            return best_action, q_values.numpy()

    def train_model(self, num_episodes=1000):
        for episode in range(num_episodes):
            list1 = torch.randint(1, 10, (1, self.list_size)).float()
            list2 = torch.randint(1, 10, (1, self.list_size)).float()
            state = (list1, list2)

            # Alternate training between the two groups
            for group in [1, 2]:
                for t in range(5):  # Perform fewer steps per group per episode
                    action = self.choose_action(state, action_group=group)
                    reward = self.reward_calculation(action, list1[0].tolist(), list2[0].tolist())
                    next_state = state

                    self.store_experience(state, action, reward, next_state, action_group=group)
                    self.train(group)

            self.decay_epsilon()

            if episode % 100 == 0:
                print(f"Episode {episode}, Epsilon: {self.epsilon:.2f}")

        print("Training completed.")

# Example usage
list_size = 5
action_size = 4  # Updated action size to include all 4 actions
agent = DQNAgent(list_size, action_size)

# Train the agent
agent.train_model(num_episodes=1000)

# Evaluate the trained model
list1 = [1, 2, 3, 4, 5]
list2 = [5, 6, 4, 6, 2]

# Choose from group 1 actions
best_action, q_values = agent.evaluate(list1, list2, action_group=1)
actions = ["SumOfList", "MultiplicationOfList", "SubtractOfList", "DivisionOfList"]
print(f"Best Action (Group 1): {actions[best_action]}")
print(f"Q-values: {q_values}")

# Choose from group 2 actions
best_action, q_values = agent.evaluate(list1, list2, action_group=2)
print(f"Best Action (Group 2): {actions[best_action]}")
print(f"Q-values: {q_values}")
