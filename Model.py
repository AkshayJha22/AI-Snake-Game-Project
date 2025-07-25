import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = self.Linear2(x)

        return x
    
    def save(self, file_name = 'model.pth'):
        model_folder_path = './mdeol'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:

    def __init__(self, model, lr , gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma

        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)

        self.criterion = nn.MSELoss() # Mean Square Error Loss Function

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.long)
        reward = torch.tensor(reward, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)

        if len(state.shape) == 1:

            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, )

        # predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_New = reward[idx]
            if not done[idx]:
                Q_New = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_New

        # Q_new = R + Y * max(Next_Predicted_Q_Value) -> only do if not done
        # pred.clone()
        # preds[argamx(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred) # target = Q_New, pred = Q
        loss.backward() # back propagation

        self.optimizer.step()

