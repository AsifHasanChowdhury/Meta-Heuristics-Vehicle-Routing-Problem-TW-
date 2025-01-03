import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import FitnessFunction as ff
import CrossOver as co
import Mutation as mo



# finalChild1 = []
# finalChild2 = []
# # Neural network definition
# class ActionValueNetwork(nn.Module):
#     def __init__(self, list_size, action_size):
#         super(ActionValueNetwork, self).__init__()
#         self.fc1 = nn.Linear(list_size * 2, 128)  # Input: concatenated lists
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, action_size)  # Output: Q-values for each action

#     def forward(self, list1, list2):
#         x = torch.cat((list1, list2), dim=1)  # Concatenate the two lists
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         q_values = self.fc3(x)  # Predict Q-values for actions
#         return q_values

# # Reward calculation function
# def rewardCalculation(action, list1, list2, instance, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0):
#     if action == 0:  # SumOfList
#         ch1, ch2 = co.cx_partially(list1,list2)
#         reward1 = ff.evaluate_individual(ch1, instance, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0)
#         reward2 = ff.evaluate_individual(ch2, instance, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0)
#         return reward1+reward2
    
#     elif action == 1:  # MultiplicationOfList
#         ch1, ch2 = co.order_crossover(list1,list2)
#         #reward = abs(sum([a * b for a, b in zip(list1, list2)]))  # Higher product sum is better
#         reward1 = ff.evaluate_individual(ch1, instance, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0)
#         reward2 = ff.evaluate_individual(ch2, instance, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0)
#         return reward1+reward2
    
#     else:
#         reward1 = ff.evaluate_individual(ch1, instance, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0)
#         reward2 = ff.evaluate_individual(ch2, instance, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0)
#         return reward1+reward2


# # Epsilon-greedy action selection
# def choose_action(state, epsilon, model):
#     list1, list2 = state
#     if random.random() < epsilon:
#         return random.randint(0, action_size - 1)  # Random action
#     else:
#         with torch.no_grad():
#             q_values = model(list1, list2)
#             return torch.argmax(q_values).item()  # Best action




# criterion = nn.MSELoss()
# replay_memory = []


# def ReinforcementDriverMethod(kid1,kid2,instance, unit_cost,init_cost, wait_cost, delay_cost):
    
#     # Hyperparameters
#     list_size = 0  # Length of each list
#     action_size = 3  # Number of actions: SumOfList, MultiplicationOfList
#     gamma = 0.99  # Discount factor
#     epsilon = 1.0  # Exploration rate
#     epsilon_decay = 0.995
#     epsilon_min = 0.1
#     learning_rate = 0.001
#     batch_size = 32
#     memory_size = 10000
#     num_episodes = 1000  # Number of episodes


# # Training loop
#     for episode in range(num_episodes):
#         # Generate random lists
#         # list1 = torch.randint(1, 10, (1, list_size)).float()
#         # list2 = torch.randint(1, 10, (1, list_size)).float()

#         list1 = kid1
#         list2 = kid2
#         state = (list1, list2)
        
#         for t in range(10):  # Number of steps per episode
#             # Choose an action
#             action = choose_action(state, epsilon, model)
            
#             # Perform the action and compute the reward
#             reward = rewardCalculation(action, list1[0].tolist(), list2[0].tolist(),instance, unit_cost,init_cost, wait_cost, delay_cost)
            
#             # Next state (same as current state in this setup)
#             next_state = state
            
#             # Store experience in replay memory
#             replay_memory.append((state, action, reward, next_state))
#             if len(replay_memory) > memory_size:
#                 replay_memory.pop(0)
            
#             # Sample a batch for training
#             if len(replay_memory) >= batch_size:
#                 batch = random.sample(replay_memory, batch_size)
#                 batch_states, batch_actions, batch_rewards, batch_next_states = zip(*batch)
                
#                 # Prepare tensors
#                 batch_list1 = torch.cat([s[0] for s in batch_states])
#                 batch_list2 = torch.cat([s[1] for s in batch_states])
#                 batch_next_list1 = torch.cat([s[0] for s in batch_next_states])
#                 batch_next_list2 = torch.cat([s[1] for s in batch_next_states])
#                 batch_actions = torch.tensor(batch_actions)
#                 batch_rewards = torch.tensor(batch_rewards).float()
                
#                 # Compute target Q-values
#                 with torch.no_grad():
#                     target_q_values = batch_rewards + gamma * torch.max(
#                         model(batch_next_list1, batch_next_list2), dim=1
#                     )[0]
                
#                 # Compute current Q-values
#                 current_q_values = model(batch_list1, batch_list2)
#                 current_q_values = current_q_values.gather(1, batch_actions.unsqueeze(1)).squeeze()
                
#                 # Compute loss and update network
#                 loss = criterion(current_q_values, target_q_values)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
        
#         # Decay epsilon
#         epsilon = max(epsilon * epsilon_decay, epsilon_min)

#         # Logging progress
#         if episode % 100 == 0:
#             print(f"Episode {episode}, Epsilon: {epsilon:.2f}, Last Reward: {reward}")

#     print("Training completed.")

#     # best_action, q_values = evaluate_model(model, list1, list2)
#     # actions = ["SumOfList", "MultiplicationOfList"]
#     # actions = ActionList

#     # Evaluate the model
# def makeDecision(list1, list2):
#     with torch.no_grad():
#         list1_tensor = torch.tensor([list1], dtype=torch.float32)
#         list2_tensor = torch.tensor([list2], dtype=torch.float32)
#         q_values = model(list1_tensor, list2_tensor)
#         best_action = torch.argmax(q_values).item()
#         return best_action, q_values.numpy()

# # Example lists for evaluation
# list1 = [1, 2, 3, 4, 5]
# list2 = [5, 6, 4, 6, 2]


# # Get the best action and Q-values

# # print(f"Best Action: {actions[best_action]}")
# # print(f"Q-values: {q_values}")


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
        self.actions = ["cx_partially", "order_crossover", "swap_mutation", "inverse_mutation"]
        self.action_groups = {
            1: [0, 1],  # Group 1: SumOfList, MultiplicationOfList
            2: [2, 3],  # Group 2: SubtractOfList, DivisionOfList
        }


    def reward_calculation(self, action, list1, list2,instance, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0):
        
        if action == 0:  # SumOfList
            ch1, ch2 = co.cx_partially(list1,list2)
            reward1 = ff.evaluate_individual(ch1, instance, unit_cost, init_cost, wait_cost, delay_cost)
            reward2 = ff.evaluate_individual(ch2, instance, unit_cost, init_cost, wait_cost, delay_cost)
            return reward1+reward2

        elif action == 1:  # MultiplicationOfList
            ch1, ch2 = co.order_crossover(list1,list2)
            #reward = abs(sum([a * b for a, b in zip(list1, list2)]))  # Higher product sum is better
            reward1 = ff.evaluate_individual(ch1, instance, unit_cost, init_cost, wait_cost, delay_cost)
            reward2 = ff.evaluate_individual(ch2, instance, unit_cost, init_cost, wait_cost, delay_cost)
            return reward1+reward2
        
        elif action == 2:
            mc1 = mo.swap_mutation(list1)
            mc2 = mo.swap_mutation(list2)
            reward1 = ff.evaluate_individual(mc1, instance, unit_cost, init_cost, wait_cost, delay_cost)
            reward2 = ff.evaluate_individual(mc2, instance, unit_cost, init_cost, wait_cost, delay_cost)
            return reward1+reward2
        
        elif action == 3:
            mc1 = mo.inverse_mutation(list1)
            mc2 = mo.inverse_mutation(list2)
            reward1 = ff.evaluate_individual(mc1, instance, unit_cost, init_cost, wait_cost, delay_cost)
            reward2 = ff.evaluate_individual(mc2, instance, unit_cost, init_cost, wait_cost, delay_cost)
            return reward1+reward2
        # else:
        #     reward1 = ff.evaluate_individual(list1, instance, unit_cost, init_cost, wait_cost, delay_cost)
        #     reward2 = ff.evaluate_individual(list2, instance, unit_cost, init_cost, wait_cost, delay_cost)
        #     return reward1+reward2


    # def choose_action(self, state):
    #     list1, list2 = state
    #     if random.random() < self.epsilon:
    #         return random.randint(0, self.action_size - 1)
    #     else:
    #         with torch.no_grad():
    #             q_values = self.model(list1, list2)
    #             return torch.argmax(q_values).item()

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



    # def store_experience(self, state, action, reward, next_state):
    #     self.replay_memory.append((state, action, reward, next_state))
    #     if len(self.replay_memory) > self.memory_size:
    #         self.replay_memory.pop(0)

    # def train(self):
    #     if len(self.replay_memory) < self.batch_size:
    #         return

    #     batch = random.sample(self.replay_memory, self.batch_size)
    #     batch_states, batch_actions, batch_rewards, batch_next_states = zip(*batch)

    #     batch_list1 = torch.cat([s[0] for s in batch_states])
    #     batch_list2 = torch.cat([s[1] for s in batch_states])
    #     batch_next_list1 = torch.cat([s[0] for s in batch_next_states])
    #     batch_next_list2 = torch.cat([s[1] for s in batch_next_states])
    #     batch_actions = torch.tensor(batch_actions)
    #     batch_rewards = torch.tensor(batch_rewards).float()

    #     with torch.no_grad():
    #         target_q_values = batch_rewards + self.gamma * torch.max(
    #             self.model(batch_next_list1, batch_next_list2), dim=1
    #         )[0]

    #     current_q_values = self.model(batch_list1, batch_list2)
    #     current_q_values = current_q_values.gather(1, batch_actions.unsqueeze(1)).squeeze()

    #     loss = self.criterion(current_q_values, target_q_values)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    # def decay_epsilon(self):
    #     self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    # def evaluate(self, list1, list2,action_group=None):
    #     with torch.no_grad():
    #         list1_tensor = torch.tensor([list1], dtype=torch.float32)
    #         list2_tensor = torch.tensor([list2], dtype=torch.float32)
    #         q_values = self.model(list1_tensor, list2_tensor)
    #         available_actions = self.action_groups.get(action_group, list(range(self.action_size)))
    #         q_values_filtered = torch.tensor([q_values[0][a] for a in available_actions])
    #         best_action = available_actions[torch.argmax(q_values_filtered).item()]
    #         return best_action, q_values.numpy()

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





    def train_model(self, num_episodes, kid1,kid2, instance, 
                    unit_cost, init_cost, wait_cost, 
                    delay_cost,user_choice_group):
        
        for episode in range(num_episodes):

            list1 =  torch.tensor(kid1, dtype=torch.float32).unsqueeze(0)
            list2 =  torch.tensor(kid2, dtype=torch.float32).unsqueeze(0)
            state = (list1, list2)


# Determine the group to train based on user choice
            groups_to_train = (
                [user_choice_group] if user_choice_group in [1, 2] else [1, 2]
            )

            for group in groups_to_train:
                for t in range(10):
                    action = self.choose_action(state)
                    reward = self.reward_calculation(action, kid1, kid2,instance, unit_cost, init_cost, wait_cost, delay_cost)
                    next_state = state

                    self.store_experience(state, action, reward, next_state,action_group=group)
                    self.train(group)

            self.decay_epsilon()

            if episode % 100 == 0:
                print(f"Episode {episode}, Epsilon: {self.epsilon:.2f}")

        print("Training completed.")









# list_size = 5
# action_size = 2
# agent = DQNAgent(list_size, action_size)

# # # Evaluate the trained model
# list1 = [1, 1, 1, -4, 5]
# list2 = [0, 5, -4, -5, -6]
# agent.train_model(1,list1,list2)

# best_action, q_values = agent.evaluate(list1, list2)
# actions = ["SumOfList", "MultiplicationOfList"]
# print(f"Best Action: {actions[best_action]}")
# print(f"Q-values: {q_values}")
