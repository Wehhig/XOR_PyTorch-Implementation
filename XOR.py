import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#hiperparametry
SEED = 0
POWIELENIE = 10
LR = 0.5
EPOCHS = 4000
TRAIN_SPLIT = 0.8
H1 = 2

#ustawienia
np.random.seed(SEED)
torch.manual_seed(SEED)

def f(x):
    return torch.sigmoid(x)

#dane xor
X_base = np.array([
    [-1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0]
], dtype=np.float32)

t_base = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
], dtype=np.float32)

X_all = np.repeat(X_base, POWIELENIE, axis=0)
t_all = np.repeat(t_base, POWIELENIE, axis=0)

#podział na train/test
n = len(X_all)
n_tr = int(TRAIN_SPLIT * n)

idx = np.random.permutation(n)
X_tr = torch.tensor(X_all[idx[:n_tr]], dtype=torch.float32)
t_tr = torch.tensor(t_all[idx[:n_tr]], dtype=torch.float32)

X_te = torch.tensor(X_all[idx[n_tr:]], dtype=torch.float32)
t_te = torch.tensor(t_all[idx[n_tr:]], dtype=torch.float32)

#model
class SiecXOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.warstwa1 = nn.Linear(2, H1)
        self.warstwa2 = nn.Linear(H1, 1)

    def forward(self, x):
        x2 = f(self.warstwa1(x))
        y2 = f(self.warstwa2(x2))
        return y2

model = SiecXOR()

#trening
optimizer = optim.SGD(model.parameters(), lr=LR)
kryterium = nn.MSELoss()

hist_tr = []
hist_te = []

for ep in range(EPOCHS):
    optimizer.zero_grad()
    y_pred_tr = model(X_tr)
    loss_tr = kryterium(y_pred_tr, t_tr)
    loss_tr.backward()
    optimizer.step()

    with torch.no_grad():
        y_pred_te = model(X_te)
        loss_te = kryterium(y_pred_te, t_te)

    hist_tr.append(loss_tr.item())
    hist_te.append(loss_te.item())

    if ep % 500 == 0:
        print(f"epoka {ep}: E_tr={loss_tr.item():.4f}, E_te={loss_te.item():.4f}")

print("\nuczenie zakończone.\n")

#test xor
print("test na 4 punktach XOR:")
with torch.no_grad():
    X_test4 = torch.tensor(X_base, dtype=torch.float32)
    t_test4 = torch.tensor(t_base, dtype=torch.float32)
    y_out = model(X_test4)

    for i in range(4):
        x_i = X_test4[i].tolist()
        t_i = t_test4[i].item()
        y_i = y_out[i].item()
        y_bin = 1 if y_i >= 0.5 else 0
        print(f"{x_i}  t={t_i}  y={y_i:.4f} -> {y_bin}")

#wynik błędu
plt.figure()
plt.plot(hist_tr, label="E_train")
plt.plot(hist_te, label="E_test")
plt.xlabel("epoka")
plt.ylabel("błąd (MSE)")
plt.grid(True)
plt.legend()
plt.savefig("nazwa_plik.png", dpi=300, bbox_inches='tight')
plt.show()


