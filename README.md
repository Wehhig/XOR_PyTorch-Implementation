Made with ☕ + PyTorch

# 🧠 XOR Gate with a Tiny MLP (PyTorch)

A small project showing how a **multi-layer perceptron (MLP)** can learn the **XOR logic gate** — a classic example of a **non-linearly separable** problem.

✨ **Key idea:** a single perceptron can’t solve XOR, but a **2–2–1** network can.

---

## 🎯 What this project does

✅ Builds a simple MLP in **PyTorch**  
✅ Trains it to learn XOR using **Sigmoid + MSELoss + SGD**  
✅ Uses a formal **80% train / 20% test split** (data is repeated only to make the split possible)  
✅ Prints training progress and final predictions  
✅ Generates a **loss curve** for a report

---

## 🖼️ Screenshots

**Loss curve (train vs test):**  
![Loss curve (train/test)](assets/1.png)

**Network diagram (MLP 2–2–1):**  
![MLP 2-2-1 diagram](assets/2.png)

---

## 📊 Results (example run)

After training, the network correctly classifies all XOR cases (threshold at **0.5**):

| x1 | x2 | target (t) | output (y) | class |
|---:|---:|-----------:|-----------:|------:|
| -1 | -1 | 0 | 0.0457 | 0 |
| -1 |  1 | 1 | 0.9630 | 1 |
|  1 | -1 | 1 | 0.9620 | 1 |
|  1 |  1 | 0 | 0.0434 | 0 |

Training/testing error drops close to zero (checkpoints):

- epoch 0: `E_tr=0.2457`, `E_te=0.2818`  
- epoch 1000: `E_tr=0.0278`, `E_te=0.0370`  
- epoch 3500: `E_tr=0.0020`, `E_te=0.0022`

---

## 🧩 Dataset (XOR)

Inputs follow the convention from the lecture slides:

- **0 → -1**
- **1 →  1**

Truth table:

| x1 | x2 | XOR |
|---:|---:|---:|
| -1 | -1 |  0 |
| -1 |  1 |  1 |
|  1 | -1 |  1 |
|  1 |  1 |  0 |

---

## 🏗️ Model Architecture

**MLP 2–2–1** (minimal XOR network):

- 2 inputs  
- 2 hidden neurons (sigmoid)  
- 1 output neuron (sigmoid)

> Bias is handled automatically by `nn.Linear`.

---

## ⚙️ Hyperparameters

All hyperparameters are placed at the top of the script for easy tweaking:

- `SEED` – reproducible results (optional)
- `POWIELENIE` – repeats XOR samples to enable 80/20 split
- `LR` – learning rate
- `EPOCHS` – number of training epochs
- `TRAIN_SPLIT` – train/test ratio
- `H1` – hidden layer size (2 for minimal XOR)

---

## ⭐ Ideas to extend

- try different `LR` values and compare convergence speed
- change hidden size `H1` (e.g., 4) and observe how learning changes
- switch to `BCELoss` and compare with `MSELoss`
