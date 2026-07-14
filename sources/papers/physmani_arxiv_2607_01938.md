# PhysMani: Physics-principled 3D World Model for Dynamic Object Manipulation（arXiv:2607.01938）

> 来源归档（ingest）

- **标题：** PhysMani: Physics-principled 3D World Model for Dynamic Object Manipulation
- **类型：** paper
- **venue：** ECCV 2026
- **arXiv：** <https://arxiv.org/abs/2607.01938>（PDF：<https://arxiv.org/pdf/2607.01938v1>）
- **代码：** <https://github.com/vLAR-group/PhysMani>（当前为项目 landing page；代码/数据/权重尚未发布）
- **机构：** vLAR Group，香港理工大学（PolyU）；Astribot（中国）
- **作者：** Peng Yun, Shouwang Huang, Hao Li, Jinxi Li, Jianan Wang, Bo Yang（通讯）
- **入库日期：** 2026-07-14
- **一句话说明：** 面向 **非结构化 3D 环境中快速动态目标操作** 的轻量框架：以 **在线优化的无散度 3D Gaussian 速度场世界模型** 预测物理可信的未来场景动态，再以 **可学习 token 交叉注意力** 将预测注入 **3DFA 策略骨干**；提出 **PhysMani-Bench（16 任务）** 并在仿真与 **Astribot S1** 真机动态任务上显著优于 3D 策略与 π₀.₅ 等强基线。

## 摘要级要点

- **问题：** VLA 与视频世界模型在 **3D 几何显式性**、**物理有意义预测** 与 **低延迟推理** 三方面不足，难以支撑接球、向移动容器投物、向旋转台面放盘等 **时间敏感动态操作**。
- **双模块并行：**
  1. **Physics-principled 3D Gaussian world model：** 在 FreeGave 基础上 **移除辅助形变场与离线优化**，改为 **流式 RGB-D 在线优化** 学习 **无散度 per-Gaussian 速度场**；单轮约 **200 ms**（RTX 4090，T=50）。
  2. **Future-aware action policy：** 以 **3D FlowMatch Actor（3DFA）** 为骨干；对稀疏 3D 点云每点 KNN 检索邻近 Gaussian，经 **可学习 query token L** 的 cross-attention 融合 **六维基本速度分量 D_t**，再 Rectified Flow 去噪预测末端 keypose。
- **PhysMani-Bench：** 自 RLBench 扩展 **8 组 × 常速/高速 = 16 任务**（beat buzz、insert peg、drop to hoop、pick moving cube、push button、deposit rubbish、place/remove cup on rack 等）；每任务 **160** 条脚本专家轨迹（100/20/40 划分）；**单模型** 跨 16 任务训练。
- **主要数字（论文报告）：** 仿真 **Mean SR 45.9%**（次优 3DFA **37.8%**；π₀.₅ **8.3%**）；未来帧 **PSNR 26.90**（第 1 帧，优于 FreeGave **19.47**）；真机 4 任务 **Mean SR 62.5%** vs 3DFA **45.3%**；推理 **272.8 ms/keyframe**（世界模型 ~200 ms 并行）。

## 核心论文摘录（MVP）

### 1) 动机：动态操作需要 3D + 物理 + 低延迟

- **链接：** <https://arxiv.org/abs/2607.01938> Abstract；§1 Introduction
- **摘录要点：** 现有 VLA / 视频 WM 缺乏 **显式 3D 场景几何**、生成帧 **物理语义弱**、大模型链 **推理延迟高**；动态交互须 **快速准确预见未来** 并 **在 3D 空间精确执行**。
- **对 wiki 的映射：**
  - [PhysMani（动态操作 3D 世界模型）](../../wiki/entities/paper-physmani-dynamic-manipulation-world-model.md) — 总览与 Mermaid。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 动态目标操作子问题。

### 2) 世界模型：Canonical 3DGS + 无散度速度场 + 在线优化

- **链接：** §3.2；Algorithm 1；Fig. 2
- **摘录要点：** t=0 用多相机 RGB-D 初始化 canonical Gaussians；**f_vel MLP** 预测六基本速度分量 V_t，经基向量 B(g_t) 组合为 **v(g_t,t)**（FreeGave 证明 **divergence-free**）；每帧接收新 RGB-D → 速度驱动 Gaussian 推进 → **T=50** 次梯度优化位置/朝向与 f_vel，再 **T'=7** 冻结速度网微调外观；**静态计算图复用** 降低 CUDA 启动开销。
- **对 wiki 的映射：**
  - [PhysMani](../../wiki/entities/paper-physmani-dynamic-manipulation-world-model.md) — 世界模型三模块表。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — **3DGS 物理速度场** 相对 **2D 视频扩散 WM** 的对照轴。

### 3) 策略：KNN 邻域 + 可学习 token 融合未来动态

- **链接：** §3.3；Fig. 3
- **摘录要点：** 当前观测经 3DFA 编码为 **4096×120** 视觉 token；世界模型输出 **H=30000** Gaussian 的 **D_t∈R^{H×6}**；每 3D 点 KNN 取 K 邻近 Gaussian，相对偏移 ΔP 与速度 D̂_t 经 MLP → **learnable L** 作 query 的 cross-attention → 动态 token D̃_t 加至视觉 token → Rectified Flow 预测 **J 步 keypose**（pos/rot/open）。
- **对 wiki 的映射：**
  - [PhysMani](../../wiki/entities/paper-physmani-dynamic-manipulation-world-model.md) — 动态注入六步与消融 (1)(2)。
  - [Behavior Cloning](../../wiki/methods/behavior-cloning.md) — 3DFA / 扩散–流匹配 IL 语境。

### 4) PhysMani-Bench 与单模型跨任务评测

- **链接：** §4；Fig. 4；§5.1 Table 1–2
- **摘录要点：** 移动目标速度与 Franka 末端上限（~2 m/s）可比；**4 固定相机** 同 RLBench；对比 Act3D、3DDA、3DFA、3DFA-OF（+光流）、ManiGaussian、π₀.₅（LoRA 微调）；PhysMani 在 **Drop to Hoop、Pick Cube** 等高速任务增益最大；相对 FreeGave **3.0× 帧率** 且 PSNR 更高。
- **对 wiki 的映射：**
  - [PhysMani](../../wiki/entities/paper-physmani-dynamic-manipulation-world-model.md) — Benchmark 与 SR 表。
  - [Manipulation](../../wiki/tasks/manipulation.md) — RLBench 风格静态→动态 benchmark 扩展语境。

### 5) 真机 Astribot S1 与效率

- **链接：** §5.2 Table 3；§5.5 Table 5；Fig. 7
- **摘录要点：** **Astribot S1** 双指臂 + **4× RGB-D（240×320）**；**5 Hz** 控制 + action buffer temporal ensemble；4 任务（传送带 pick/place、旋转架 place/remove）；每法 **单模型** 跨 4 任务；PhysMani **62.5%** mean SR；训练 **53 h**，推理 **272.8 ms/frame**，显存 **1.0 GB**。
- **对 wiki 的映射：**
  - [PhysMani](../../wiki/entities/paper-physmani-dynamic-manipulation-world-model.md) — 真机任务与效率节。
  - [vLAR-group/PhysMani 仓库归档](../repos/vlar_group_physmani.md)

## BibTeX

```bibtex
@inproceedings{yun2026physmani,
  title     = {{PhysMani}: Physics-principled 3D World Model for Dynamic Object Manipulation},
  author    = {Yun, Peng and Huang, Shouwang and Li, Hao and Li, Jinxi and Wang, Jianan and Yang, Bo},
  booktitle = {European Conference on Computer Vision},
  year      = {2026},
  eprint    = {2607.01938},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url       = {https://arxiv.org/abs/2607.01938}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-physmani-dynamic-manipulation-world-model.md`](../../wiki/entities/paper-physmani-dynamic-manipulation-world-model.md)
- 代码归档：[`sources/repos/vlar_group_physmani.md`](../repos/vlar_group_physmani.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[Kairos（视频 WAM 栈）](../../wiki/entities/paper-kairos-native-world-model-stack.md)、[MINT（3D IL 泛化）](../../wiki/entities/paper-mint-vla.md)
