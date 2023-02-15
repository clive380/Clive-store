import torch
import torch.nn


class trainModel1(torch.nn.Module):
    def __init__(self, num_steps):
        super(trainModel1, self).__init__()
        self.num_steps = num_steps
        self.lstm = torch.nn.LSTM(num_steps, 10)
        self.linear1 = torch.nn.Linear(10, 8)
        self.linear2 = torch.nn.Linear(8, 6)
        self.linear3 = torch.nn.Linear(6, 4)
        self.linear4 = torch.nn.Linear(4, 1)
        self.F1 = torch.nn.Tanh()
        self.F2 = torch.nn.ReLU()

    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x, hc = self.lstm(x)
        x = self.F1(x)
        x = self.F2(self.linear1(x))
        x = self.F2(self.linear2(x))
        x = self.F2(self.linear3(x))
        x = self.F2(self.linear4(x))
        return x
