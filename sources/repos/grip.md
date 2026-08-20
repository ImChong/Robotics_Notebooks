# GRIP（Ground Reaction Inertial Poser）

> 来源归档

- **标题：** GRIP — Official implementation
- **类型：** repo
- **来源：** Ryosuke Hori 等（CMU / Keio）；共同作者含 [Zhengyi Luo](https://github.com/ZhengyiLuo)（PHC / SimXR 生态）
- **链接：** <https://github.com/RyosukeHori/GRIP>
- **论文：** <https://arxiv.org/abs/2603.16233>（CVPR 2026）
- **项目页：** <https://ryosukehori.github.io/grip-project/>
- **配套数据集：** [RyosukeHori/PRISM](https://github.com/RyosukeHori/PRISM) — 见 [`prism-dataset.md`](./prism-dataset.md)
- **入库日期：** 2026-08-20
- **一句话说明：** 稀疏 IMU + 鞋垫压力的人体 MoCap：**KinematicsNet**（PyTorch + articulate）+ **DynamicsNet**（Hydra + Isaac Gym + rl_games + PPO）；含 PRISM 预处理、评测与 aitviewer 可视化。

---

## 仓库布局（README）

| 目录 | 说明 |
|------|------|
| `kinematics_net/` | KinematicsNet 训练 / 推理 |
| `dynamics_net/` | DynamicsNet（Isaac Gym 仿真 + rl_games） |
| `data_process/` | 原始 PRISM → KinematicsNet / DynamicsNet 数据集 |
| `visualization/` | aitviewer 可视化各阶段 |
| `evaluation/` | MPJPE / MPJRE / Foot Slide / FP / GRF Error |
| `scripts/` | shell 入口（项目根目录执行） |
| `data/` | SMPL 模型 + 预处理张量（需自行下载） |

---

## 依赖与数据（2026-08-20 核查）

| 项 | 说明 |
|----|------|
| Python | 3.8；PyTorch 2.1.1 + CUDA 12.1 wheel（`scripts/setup_grip.sh`） |
| 仿真 | **Isaac Gym Preview 4**（DynamicsNet；需 NVIDIA 开发者站下载） |
| SMPL | `data/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl` |
| 预处理数据 | Google Form 下载 ~4.5 GB → `data/preprocessed/` |
| 预训练 | `output/kinematics_net/models/best_model.pt`、`output/dynamics_net/Humanoid.pth`（Drive / output.zip） |

---

## 主要入口

```bash
conda env create -f environment.yml --solver=libmamba && conda activate grip
bash scripts/setup_grip.sh
# DynamicsNet 另需 pip install -e isaacgym/python

python data_process/kinematics_dataset.py
bash scripts/kinematics_train.sh
bash scripts/kinematics_inference.sh

python data_process/dynamics_dataset.py
bash scripts/dynamics_train.sh
bash scripts/dynamics_test.sh

bash scripts/evaluate.sh
```

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-grip](../../wiki/entities/paper-grip.md) | 论文实体、流程图与时序图 |
| [phc](../../wiki/entities/phc.md) | DynamicsNet 奖励参考 PHC（AMP + imitation + energy） |
| [prism-dataset.md](./prism-dataset.md) | 原始多模态采集数据 |
