# 🐍 AI Snake Game

An AI-powered Snake game built with **PyTorch**, **Pygame**, and **Matplotlib**.  
In this project, an AI agent learns to play Snake autonomously using deep reinforcement learning.

---

## 📌 Features

- Classic Snake game implemented in **Pygame**
- AI agent using **Deep Q-Learning (DQN)** built with **PyTorch**
- Real-time training metrics and plots with **Matplotlib**
- Modular, easy-to-read code

---

## 🚀 Run It

```bash
pip install -r requirements.txt
python Agent.py
```

Training opens the game window and a live score plot. Each time the agent sets a new record score, its weights
are saved to `model/model.pth`.
