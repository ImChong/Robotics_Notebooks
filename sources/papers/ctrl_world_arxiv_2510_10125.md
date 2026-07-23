# Ctrl-World: A Controllable Generative World Model for Robot Manipulation（arXiv:2510.10125 / ICLR 2026）

> 来源归档（ingest）

- **标题：** Ctrl-World: A Controllable Generative World Model for Robot Manipulation
- **类型：** paper / controllable world model / policy evaluation / policy improvement / multi-view
- **arXiv：** <https://arxiv.org/abs/2510.10125>（PDF：<https://arxiv.org/pdf/2510.10125.pdf>；v3 修订至 2026-03-01）
- **会议：** ICLR 2026
- **项目页：** <https://ctrl-world.github.io/>
- **代码：** <https://github.com/Robert-gyj/Ctrl-World>
- **权重：** <https://huggingface.co/yjguo/Ctrl-World>
- **作者：** Yanjiang Guo*、Lucy Xiaoyang Shi*、Jianyu Chen、Chelsea Finn（* 同等贡献）
- **机构：** 斯坦福大学（Stanford）、清华大学（Tsinghua）
- **入库日期：** 2026-07-23
- **一句话说明：** 从 SVD 初始化的 **可控多视角** 动作条件世界模型：帧级动作条件 + 位姿条件记忆检索，在 DROID 上训出可与现代 VLA（含腕部相机）做 **policy-in-the-loop** 想象 rollout；用于无真机 rollout 的策略排序，以及合成成功轨迹 SFT（π₀.₅ 新指令成功率 **38.7%→83.4%**，约 **+44.7 pt**）。

## 开源状态（项目页 + 仓库核查，2026-07-23）

- **已开源：** 项目页 Code → [`Robert-gyj/Ctrl-World`](https://github.com/Robert-gyj/Ctrl-World)（**MIT**）+ HF [`yjguo/Ctrl-World`](https://huggingface.co/yjguo/Ctrl-World)；含 replay / 键盘 / π₀.₅ 交互脚本与 DROID 训练管线；依赖 SVD + CLIP +（可选）openpi。

## 摘要级要点

- **瓶颈：** 通用策略评估与纠错依赖大量真机 rollout；已有动作条件 WM 多为单第三人称、控制粒度粗、长时一致性差，难以接入现代多视角 VLA。
- **三件套：** (1) **多视角联合预测**（第三人称 + 腕部）；(2) **帧级动作 / 位姿条件**（空间 transformer 内 frame-wise cross-attention）；(3) **位姿条件记忆检索**（稀疏历史帧 + 位姿锚定）。
- **数据 / 训练：** DROID **~95k** 轨迹 / **564** 场景（含成功与失败）；SVD **1.5B** 初始化；2×8 H100，约 **2–3 天**；分辨率 **192×320**，历史 **7** 帧，动作块 **15** 步（约 1 s）。
- **下游：** 想象 rollout 与真机 **指令跟随** 排名对齐；合成成功轨迹 SFT 提升未见指令 / 物体上的指令跟随。

## 核心论文摘录（MVP）

### 1) Policy–WM 闭环公式

- **链接：** §3；Eq. (1)–(2)
- **摘录要点：** \(a_{t+1:t+H}\sim\pi(\cdot\mid o_t,l)\)，\(o_{t+1:t+H}\sim W(\cdot\mid o_t,A_t)\)，再把 \(o_{t+H}\) 回传策略，自回归想象。
- **对 wiki 的映射：**
  - [Ctrl-World](../../wiki/entities/paper-ctrl-world.md) — 闭环接口。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 策略–WM 交互范式。

### 2) 多视角 + 帧级条件 + 记忆

- **链接：** §4.1；Fig. 2–5
- **摘录要点：** 多视角拼接 token 联合预测，腕部视角显著减接触幻觉；动作转笛卡尔位姿后与历史位姿一起做帧级 cross-attention；稀疏历史 + 位姿检索稳住长时（>20 s）。
- **对 wiki 的映射：**
  - [Ctrl-World](../../wiki/entities/paper-ctrl-world.md) — 核心机制。
  - [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) — 像素级交互仿真。

### 3) 评估与合成改进

- **链接：** §4.2；§5.3–5.4；Alg. 1；Fig. 7–9
- **摘录要点：** 人偏好标注成功/失败；扰动指令或重置初态扩大搜索；筛选合成成功轨迹后 SFT；平均指令跟随提升 **44.7%**（**38.7%→83.4%**）。低层接触物理仍有 gap。
- **对 wiki 的映射：**
  - [world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md) — 虚拟评估 / 改进沙盒。
  - [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) — 同属动作条件 WM 对照（掩码 vs 低维动作）。

## BibTeX

```bibtex
@article{guo2025ctrl,
  title   = {Ctrl-World: A Controllable Generative World Model for Robot Manipulation},
  author  = {Guo, Yanjiang and Shi, Lucy Xiaoyang and Chen, Jianyu and Finn, Chelsea},
  journal = {arXiv preprint arXiv:2510.10125},
  year    = {2025}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-ctrl-world.md`](../../wiki/entities/paper-ctrl-world.md)
- 代码归档：[`sources/repos/ctrl-world.md`](../repos/ctrl-world.md)
- 项目页：[`sources/sites/ctrl-world-github-io.md`](../sites/ctrl-world-github-io.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md)、[DriftWorld](../../wiki/entities/paper-driftworld.md)、[Wan](../../wiki/entities/paper-wan-video.md)
