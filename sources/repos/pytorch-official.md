# PyTorch 官方站点与文档索引

> 来源归档（以官网叙述为准；版本号与安装命令以安装向导实时输出为准）

- **标题：** PyTorch — An open source machine learning framework
- **类型：** 官方站点 + 文档 + 教程 + 开源核心仓库
- **主页：** https://pytorch.org/
- **本地安装向导：** https://pytorch.org/get-started/locally/
- **历史版本安装：** https://pytorch.org/get-started/previous-versions
- **教程：** https://pytorch.org/tutorials/
- **稳定版文档：** https://pytorch.org/docs/stable/index.html
- **核心代码：** https://github.com/pytorch/pytorch
- **基金会：** https://pytorch.org/foundation/（站点首页「Join PyTorch Foundation」入口；治理与成员权益以基金会页为准）
- **入库日期：** 2026-05-15
- **一句话说明：** 以 **Python 优先、动态图（eager）友好** 为主线的开源深度学习框架；官网强调 **研究与生产衔接**（eager / graph、TorchScript、TorchServe）、**可扩展分布式训练**（`torch.distributed`）、**云厂商快速上手** 与 **周边生态**（如 Captum、PyTorch Geometric、skorch）；并持续通过 **ExecuTorch** 等方向扩展 **边端推理** 叙事。
- **沉淀到 wiki：** [PyTorch](../../wiki/entities/pytorch.md)

---

## 首页要点（2026-05-15 抓取归纳）

1. **发布节奏**：首页突出「PyTorch 2.x Release Blog / release notes」；具体特性以对应 release notes 为准。
2. **安装**：安装矩阵支持 **Stable / Preview (Nightly)**、**Linux / Mac / Windows**、**pip / LibTorch / Source**、**Python / C++**、**CUDA / ROCm / CPU** 等组合；官方注明 **Latest Stable PyTorch requires Python 3.10 or later**；**LibTorch 仅 C++ 侧**。
3. **生产与推理**：文案层强调 **TorchScript**、**TorchServe**；博客层推广 **ExecuTorch** 面向 **Arm CPU/NPU 等受限设备** 的实践实验材料。
4. **分布式**：将 **`torch.distributed`** 作为可扩展训练与性能优化的后端叙述。
5. **生态示例**：首页「Featured Projects」列举 **Captum**（可解释性）、**PyTorch Geometric**（图/点云/流形等非规则数据）、**skorch**（scikit-learn 风格封装）等。

---

## Get Started / Locally 页要点（归纳）

- **Stable vs Preview**：Stable 为当前测试与支持主线；Preview 为 nightly，**未完全测试与支持**。
- **前置依赖**：按包管理器准备（如 `numpy`）；GPU 路径需匹配 **CUDA / ROCm** 与驱动。
- **验证安装**：文档提供 `import torch` 与 `torch.rand`、`torch.cuda.is_available()` 等最小检查片段。
- **从源码构建**：指向 `pytorch/pytorch` 仓库 README 的 **From source** 说明。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [深度学习基础](../../wiki/concepts/deep-learning-foundations.md) | 感知与控制中的神经网络表示、优化与实现的默认教学栈之一 |
| [强化学习](../../wiki/methods/reinforcement-learning.md) | 策略/价值网络与并行 rollout 训练的主流张量后端 |
| [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md) | NVIDIA 机器人学习仿真管线与 PyTorch 生态常见组合 |
| [LeRobot](../../wiki/entities/lerobot.md) | Hugging Face 具身栈与 Transformers 生态常以 PyTorch 为训练运行时 |

---

## 对 wiki 的映射

- 新建 **`wiki/entities/pytorch.md`**：框架实体页（能力边界、与机器人训练/部署关系、互链）。
- 轻量更新 **`wiki/concepts/deep-learning-foundations.md`**、**`wiki/entities/isaac-gym-isaac-lab.md`**：补充交叉引用，避免孤岛页。

---

## 外部参考（便于复核）

- [PyTorch 官网](https://pytorch.org/)
- [Get Started（本地安装）](https://pytorch.org/get-started/locally/)
- [Tutorials](https://pytorch.org/tutorials/)
- [Documentation（stable）](https://pytorch.org/docs/stable/index.html)
- [pytorch/pytorch（GitHub）](https://github.com/pytorch/pytorch)
