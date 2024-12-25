import math
import torch
import random

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # Current state of the node
        self.parent = parent  # Parent node
        self.action = action  # Action that led to this node
        self.wins = 0  # Total score of the node
        self.visits = 0  # Number of times the node has been visited
        self.children = []  # List of child nodes
        self.available_actions = [1, 2, 3, 4]  # Example actions (can be modified)
        self.action_rewards = {action: 0 for action in self.available_actions}  # Track rewards for each action

class MCTSWithNN:
    def __init__(self, actions, initial_state, reward_model, transition_model, value_function):
        self.actions = actions  # List of available actions
        self.state = initial_state  # Initial state of the problem
        self.reward_model = reward_model  # Reward prediction model (neural network)
        self.transition_model = transition_model  # Transition model (neural network)
        self.value_function = value_function  # The function that calculates the outcome value of each action

        # Root node initialization
        self.root = Node(state=self.state)

    def uct(self, node, C=1.4):
        """ Upper Confidence Bound for Trees (UCT) formula """
        if node.visits == 0:
            return float('inf')  # Encourage exploration of unvisited nodes
        return (node.wins / node.visits) + C * math.sqrt(math.log(node.parent.visits) / node.visits)

    def select(self, node):
        """ Select the best child node based on UCT """
        best_value = -float('inf')
        best_node = None
        for child in node.children:
            uct_value = self.uct(child)
            if uct_value > best_value:
                best_value = uct_value
                best_node = child
        return best_node

    def expand(self, node):
        """ Expand the node by adding a new child node with a random action """
        for action in self.actions:
            # Simulate taking the action to create a child node
            next_state = self.simulate_transition(node.state, action)
            new_node = Node(state=next_state, parent=node, action=action)
            node.children.append(new_node)

    def simulate(self, node):
        """ Simulate the outcome for a given node """
        state = node.state
        action = node.action
        
        if state is None or action is None:
            print(f"Error: State or action is None. State: {state}, Action: {action}")
            return None, None

        state_tensor = torch.FloatTensor(state + [action])  # Combine state and action into a tensor
        next_state = self.transition_model(state_tensor).detach().numpy()

        reward = self.reward_model(state_tensor).detach().numpy()[0]

        if reward is None:
            print(f"Error: Reward is None. State: {state}, Action: {action}")
            return None, None

        # Track the reward for the action taken
        node.action_rewards[action] += reward

        return reward, next_state

    def backpropagate(self, node, reward):
        """ Backpropagate the reward through the tree """
        while node is not None:
            if reward is not None:
                node.wins += reward  # Accumulate the reward
                node.visits += 1  # Increment the visit count
            else:
                print(f"Skipping backpropagation due to None reward. Node: {node.state}")
            node = node.parent  # Move to the parent node for backpropagation

    def best_action(self, node):
        """ Select the action with the highest win rate, considering the outcome value function """
        best_action = None
        best_win_rate = -float('inf')
        
        for action, total_reward in node.action_rewards.items():
            # Incorporate the outcome value (expected long-term value) of the action
            expected_value = self.value_function(node.state, action)  # Get the expected value for the action
            win_rate = total_reward / (node.visits + 1) + expected_value  # Combine win rate and expected value
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_action = action
        return best_action

    def run(self, simulations):
        """ Run MCTS with the given number of simulations """
        for _ in range(simulations):
            node = self.root

            # Selection: Traverse the tree to find the most promising node
            while node.children:
                node = self.select(node)

            # Expansion: If the node has not been fully expanded, expand it
            if node.visits == 0:
                self.expand(node)

            # Simulation: Simulate the reward for this node
            reward, _ = self.simulate(node)
            if reward is None:
                print("Error: Simulation returned None reward")
                continue  # Skip invalid simulations

            # Backpropagation: Update the node values with the simulation reward
            self.backpropagate(node, reward)

    def simulate_transition(self, state, action):
        """ Simulate the transition to the next state after taking an action """
        state_tensor = torch.FloatTensor(state + [action])  # Combine state and action into a tensor
        next_state = self.transition_model(state_tensor).detach().numpy()
        return next_state


# Example usage:

# Dummy neural network models (you can replace these with actual trained models)
class DummyRewardModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([random.randint(100, 1000)])  # Random reward for the example

class DummyTransitionModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([x[0].item() + 1, x[1].item() + 2, x[2].item() + 3, x[3].item() + 4, x[4].item() + 5, x[5].item() + 6])  # Just for illustration

# Define your value function (which predicts the outcome value of actions)
def value_function(state, action):
    # A simple heuristic, you could replace it with a more complex function or model
    if action == 1:
        return 100  # For example, action 1 has an expected value of 100
    elif action == 2:
        return 200  # Action 2 has an expected value of 200
    elif action == 3:
        return 100000  # Action 3 has an expected value of 400
    else:
        return 1000  # Action 4 has an expected value of 1000

# Initialize models
reward_model = DummyRewardModel()
transition_model = DummyTransitionModel()

# Initial state and actions
initial_state = [1, 4, 5, 6, 8, 2]
actions = [1, 2, 3, 4]

# Run MCTS with value function
mcts_with_nn = MCTSWithNN(actions, initial_state, reward_model, transition_model, value_function)
mcts_with_nn.run(simulations=1000)

# Get the best action based on the MCTS results
best_action = mcts_with_nn.best_action(mcts_with_nn.root)
print(f"Best action to take: {best_action}")
