# SonyResearch/ace_public（Sony AI Ace 补充材料仓库）

> 来源归档

- **标题：** ace_public — Ace Supplementary Material
- **类型：** repo / supplementary / dataset / pseudo-code
- **来源：** Sony Research Inc. / Sony AI
- **链接：** <https://github.com/SonyResearch/ace_public>
- **项目页：** <https://sonyresearch.github.io/ace_public/>
- **论文：** [Nature s41586-026-10338-5](https://doi.org/10.1038/s41586-026-10338-5) — 归档见 [`sources/papers/sony_ace_nature_2026.md`](../papers/sony_ace_nature_2026.md)
- **入库日期：** 2026-09-16
- **一句话说明：** Ace 官方 **部分开源** 仓：对 elite/pro **match 后球态 CSV**、TensorFlow 风格 **SAC/FAOC/GCS/发球 GA 伪代码** 与补充视频；**不含** 训练权重、真机感知部署或 8-DOF 硬件栈。
- **沉淀到 wiki：** [`wiki/entities/paper-sony-ai-ace-table-tennis.md`](../../wiki/entities/paper-sony-ai-ace-table-tennis.md)

---

## 发布内容

| 路径 | 内容 |
|------|------|
| [`data/match_data.csv`](https://github.com/SonyResearch/ace_public/blob/main/data/match_data.csv) | 对打 post-event 球位置/速度/spin（见 [`data/Readme.md`](https://github.com/SonyResearch/ace_public/blob/main/data/Readme.md)） |
| [`pseudo_code/`](https://github.com/SonyResearch/ace_public/tree/main/pseudo_code) | 近似 Python：**train_loop**、**step_sac**、**rollout_loop** + FAOC、**serve_training**（pygad GA）、**GazeControlSystem** 伪类 |
| GitHub Pages | 完整对局、GCS、触网、发球、专家评论视频 |

## 未发布（截至 2026-09-16）

- 训练好的 **策略权重** 与赛中 **技能采样器**
- **感知栈** 部署（APS FPGA 分割、GCS TensorRT、triangulation server）
- **机器人接口**（1 kHz 轨迹执行、碰撞检测）
- **定制 8-DOF 硬件** CAD / 固件
- 可一键运行的 **端到端训练环境**

→ 开源程度：**部分** — 数据 + 结构参考；**不可** 仅凭本仓复现 Nature 级对打系统。

## 伪代码关键入口（运行时序对齐）

| 模块 | 伪代码符号 | 作用 |
|------|------------|------|
| 分布式训练 | `train_loop()` | 参数服务器 + replay + 多 table 数据集 SAC |
| 采样 worker | `rollout_loop()` | `get_task()` → `agent_policy` → **FAOC** → `env_server.step` |
| SAC 步 | `step_sac()` | 非对称 actor（噪声 obs）/ critic（真值 obs） |
| 发球 | `serve_training()` | pygad GA 离线寻优 |
| Spin 感知 | `GazeControlSystem` + `SpinEstimatorCNN` + `ContrastMaximization` | EVS 事件 → spin |

## 对 wiki 的映射

- 实体页：[Sony AI Ace（Nature 2026）](../../wiki/entities/paper-sony-ai-ace-table-tennis.md)
- 方法对照：[Table Tennis Strategy & Skill Learning](../../wiki/methods/table-tennis-strategy-skill-learning.md)
