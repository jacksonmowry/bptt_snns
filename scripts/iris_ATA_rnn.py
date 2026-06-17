#!/usr/bin/env python3

import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class AllToAllRNN(nn.Module):
    def __init__(self, n_in=4, n_hid=8, n_out=3):
        super().__init__()
        self.n_in, self.n_hid, self.n_out = n_in, n_hid, n_out
        self.N = n_in + n_hid + n_out

        # All-to-all recurrent weights within the single layer
        self.W = nn.Parameter(torch.randn(self.N, self.N) * 0.1)
        self.b = nn.Parameter(torch.zeros(self.N))
        self.out_proj = nn.Linear(self.n_hid, self.n_out)

    def forward(self, x):
        # x: (batch, seq_len, n_in)
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.N, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            full_state = torch.cat(
                [x_t, h[:, self.n_in : self.N - self.n_out], h[:, -self.n_out :]], dim=1
            )
            new_full = torch.relu(torch.matmul(full_state, self.W.T) + self.b)

            # Inject input; evolve hidden & output via recurrence
            h = torch.cat(
                [
                    h[:, : self.n_in],
                    new_full[:, self.n_in : self.N - self.n_out],
                    new_full[:, -self.n_out :],
                ],
                dim=1,
            )

        return self.out_proj(h[:, self.n_in : self.N - self.n_out])


# Data setup
data = load_iris()
X, y = StandardScaler().fit_transform(data.data), data.target
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = AllToAllRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # (batch, 1, 4)
y_t = torch.tensor(y_train, dtype=torch.long)

# Forward + Backward pass
model.train()
for e in range(10000):
    optimizer.zero_grad()
    logits = model(X_t)  # forward
    loss = criterion(logits, y_t)  # loss
    loss.backward()  # explicit backward (autograd computes gradients)
    optimizer.step()  # weight update
    print(loss)
