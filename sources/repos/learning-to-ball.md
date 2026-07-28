# Learning to Ball（xupei0610/basketball）

> 来源归档

- **标题：** Learning to Ball
- **类型：** repo
- **来源：** Pei Xu et al.（Stanford / UC Riverside / Roblox / Clemson）
- **链接：** <https://github.com/xupei0610/basketball>
- **项目页：** <https://pei-xu.github.io/basketball>
- **论文：** [arXiv:2509.22442](https://arxiv.org/abs/2509.22442) — 归档见 [`sources/papers/learning_to_ball_arxiv_2509_22442.md`](../papers/learning_to_ball_arxiv_2509_22442.md)
- **许可：** MIT
- **入库日期：** 2026-07-28
- **一句话说明：** SIGGRAPH Asia 2025 / ACM TOG 官方仓：**子技能训练与评测代码 + 预训练权重**已发布；实现基于 Composite Motion Learning 与 ICCGAN；公开发布面以原始子技能为主。
- **沉淀到 wiki：** [`wiki/entities/paper-notebook-learning-to-ball.md`](../../wiki/entities/paper-notebook-learning-to-ball.md)

---

## 核心定位

在 **Isaac Gym Preview 4** 上训练/评测物理仿真篮球角色的 **原始子技能策略**（运球、投篮、传球、接球、篮板、跑动+防守等），并作为论文「策略组合 + soft router」管线的可运行底座。

---

## 发布进度（截至 2026-07-28）

| 组件 | 状态 |
|------|------|
| 训练与评测代码（`main.py`） | ✅ |
| 子技能配置（`cfg/{dribble,shoot,catch,pass,rebound,locomotion+defend}.py`） | ✅ |
| 子技能预训练（`pretrained/` 同名目录） | ✅ |
| 依赖钉定（PyTorch 2.1.2 + Isaac Gym Pr4） | ✅ README |
| 高层 soft router / gather 独立 cfg 与预训练条目 | ⬜ 仓内清单未列 |

---

## 安装与主入口

- **环境：** `conda create --name <env> --file requirements.txt -c pytorch -c conda-forge -c nvidia`
- **依赖：** 自行从 NVIDIA 取得 **Isaac Gym Preview 4** 并 `pip` 安装
- **统一入口：** `python main.py <configure_file> --ckpt <checkpoint_dir>`

### 训练

```bash
python main.py cfg/shoot.py --ckpt ckpt_shoot
# --device N 指定 GPU（默认 0）；单卡可训
```

### 评测（预训练）

```bash
python main.py cfg/shoot.py --ckpt pretrained/shoot --test
```

同理替换 `dribble` / `catch` / `pass` / `rebound` / `locomotion+defend`。

---

## 关键目录（对齐时序图节点）

| 路径 | 角色 |
|------|------|
| `main.py` | CLI 训练 / `--test` 评测入口 |
| `env.py` | Isaac Gym 环境与奖励 |
| `models.py` | ACModel / Discriminator（含 PopArt 等） |
| `cfg/*.py` | 各子技能配置 |
| `pretrained/*` | 官方预训练子技能 |
| `ref_motion.py` / `assets/` | 参考运动与资源 |
| `requirements.txt` | Conda 依赖钉定 |

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [Composite Motion Learning](https://pei-xu.github.io/CompositeMotion)（外链） | 方法前作；README 声明实现基于该框架 |
| [SkillMimic](./skillmimic.md) | 同主题篮球长程组合对照（统一模仿 + HLC） |
| ICCGAN / AdaptNet（外链） | 对抗模仿与策略适配 |

## 对 wiki 的映射

- 实体页：[Learning to Ball](../../wiki/entities/paper-notebook-learning-to-ball.md)
- 方法页：[Hierarchical RL](../../wiki/methods/hierarchical-reinforcement-learning.md)、[Imitation Learning](../../wiki/methods/imitation-learning.md)
