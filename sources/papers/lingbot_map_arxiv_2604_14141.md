# lingbot_map_arxiv_2604_14141

> 来源归档（ingest）

- **标题：** Geometric Context Transformer for Streaming 3D Reconstruction（LingBot-Map）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2604.14141>
- **PDF（仓库内链）：** 官方 README 指向仓库内 `lingbot-map_paper.pdf`
- **项目页：** <https://technology.robbyant.com/lingbot-map>
- **代码：** <https://github.com/Robbyant/lingbot-map>
- **入库日期：** 2026-05-17
- **一句话说明：** 面向**流式**单目视频 3D 重建的前馈基础模型：用 **Geometric Context Attention（GCA）** 在单一注意力框架里显式维护 **锚点坐标/尺度接地、局部稠密几何窗口、轨迹记忆** 三类互补上下文，配合 **Paged KV Cache（FlashInfer）** 在长序列上近似常数每帧成本；报告约 **20 FPS**（518×378）与 **>10k 帧** 稳定推理，并在 Oxford Spires、7-Scenes、Tanks and Temples、ETH3D 等基准上相对既有流式与迭代优化类方法取得优势。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract）

- **链接：** <https://arxiv.org/abs/2604.14141>
- **核心贡献：** 流式 3D 重建需要**几何精度、时间一致性、计算效率**三者兼顾；作者将 SLAM 式「参考系 + 局部稠密观测 + 全局漂移校正」分解为**可学习的三类几何上下文**，用端到端注意力替代手工后端优化，在超长序列上保持紧凑流式状态。
- **对 wiki 的映射：**
  - [LingBot-Map](../../wiki/methods/lingbot-map.md)

### 2) Geometric Context Attention（GCA）三类上下文（Method 3.2）

- **链接：** <https://arxiv.org/abs/2604.14141>（§3.2）
- **核心贡献：**
  - **Anchor context**：用前 \(n\) 帧锚定**尺度与坐标系**（流式下无法像离线 VGGT 那样用全局点云归一化）。
  - **Local pose-reference window**：保留最近 \(k\) 帧的**完整 image token**，提供稠密重叠以稳定局部配准；并在窗口内施加**相对位姿损失**以鼓励局部轨迹一致。
  - **Trajectory memory**：对更早帧丢弃 image token，仅保留 **camera + anchor + register** 等少量 token（文中给出约 **6 token/帧** 量级叙述），叠加视频时序位置编码，用于**长程漂移校正**。
- **对 wiki 的映射：**
  - [LingBot-Map](../../wiki/methods/lingbot-map.md)
  - [State Estimation](../../wiki/concepts/state-estimation.md)

### 3) 训练与推理系统（§3.3–3.4 / §4）

- **链接：** <https://arxiv.org/abs/2604.14141>
- **核心贡献：** **两阶段**训练：先在短序列多视图上用全局注意力训练几何基础模型，再替换为 GCA 并以**渐进增加视点数**的课程（文中上限与显存/上下文并行相关）学习流式一致性；推理侧采用 **Paged KV cache** 与 **FlashInfer** 降低缓存重分配开销，并讨论 **keyframe 间隔** 以在超过训练最大视角数时控制 KV 增长（与官方 README 的工程说明一致）。
- **对 wiki 的映射：**
  - [LingBot-Map](../../wiki/methods/lingbot-map.md)

## BibTeX（arXiv 页 / 仓库 Citation 区）

```bibtex
@article{chen2026geometric,
  title={Geometric Context Transformer for Streaming 3D Reconstruction},
  author={Chen, Lin-Zhuo and Gao, Jian and Chen, Yihang and Cheng, Ka Leong and Sun, Yipengjing and Hu, Liangxiao and Xue, Nan and Zhu, Xing and Shen, Yujun and Yao, Yao and Xu, Yinghao},
  journal={arXiv preprint arXiv:2604.14141},
  year={2026}
}
```

## 当前提炼状态

- [x] 摘要与 §3.2 机制对齐
- [x] wiki 页面映射确认
