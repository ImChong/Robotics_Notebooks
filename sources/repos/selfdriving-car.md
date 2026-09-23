# selfdriving-car（ApexDrive AI）

> 来源归档

- **标题：** ApexDrive AI: Neuroevolution Self-Driving Simulation
- **类型：** repo
- **来源：** poojithinavolu（个人 GitHub）
- **链接：** https://github.com/poojithinavolu/selfdriving-car
- **Stars：** ~4（2026-09-23）
- **许可证：** 仓内未声明 license 文件（截至入库日）
- **入库日期：** 2026-09-23
- **一句话说明：** 纯 Python / NumPy / Pygame 的 2D 自动驾驶仿真：13 维 bumper 射线感知 + 手写 MLP，经遗传算法（Neuroevolution）在多赛道联合适应度下进化转向/油门策略，含 GUI、Human vs AI 与 champion 权重评测。
- **代码：** https://github.com/poojithinavolu/selfdriving-car（**已开源**；无 PyTorch/TensorFlow 依赖）
- **沉淀到 wiki：** 是 → [`wiki/entities/apexdrive-ai.md`](../../wiki/entities/apexdrive-ai.md)

---

## 技术要点

| 模块 | 说明 |
|------|------|
| 感知 | 9 条 bumper 射线 + 左右平衡 + 前向 clearance + 归一化速度/角速度（13D） |
| 策略 | 2 隐层 MLP（18→14），Tanh，输出连续转向与油门 |
| 优化 | 种群 50–120、多赛道同时适应度、精英保留、锦标赛选择、高斯变异 |
| 安全 | 前视减速、执行器低通、势场 repulsion |
| 赛道 | 7 条手工布局（Oval → GP → Zig-Zag → Gauntlet → Metropolis → Omega → Dragon） |

---

## 典型入口

```bash
git clone https://github.com/poojithinavolu/selfdriving-car.git
cd selfdriving-car
pip install pygame numpy

# GUI 交互 / 冠军 AI 竞速
python main.py --mode race --track 7

# 无头多赛道进化训练
python train_master.py

# 冠军权重全赛道 benchmark
python evaluate_champion.py

# 单元测试（几何、NN、GA）
python tests/test_simulation.py
```

---

## 关联

- 轻量 1/10 竞速 RL 对照：[`f1tenth_gym.md`](./f1tenth_gym.md)
- 赛车/漂移开源景观：[`racing_drift_rl_open_source_landscape.md`](../papers/racing_drift_rl_open_source_landscape.md)
