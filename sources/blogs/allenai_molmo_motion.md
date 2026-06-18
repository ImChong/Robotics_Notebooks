# MolmoMotion（Ai2 官方博客）

> 来源归档（ingest）

- **标题：** MolmoMotion: Language-guided 3D motion forecasting
- **类型：** blog + technical report
- **组织：** Allen Institute for AI (Ai2)
- **原始链接：** <https://allenai.org/blog/molmo-motion>
- **技术报告 / 论文：** <https://arxiv.org/abs/2606.18558> | <https://allenai.org/papers/molmomotion>
- **项目页：** <https://molmomotion.github.io/>
- **代码：** <https://github.com/allenai/molmo-motion>
- **模型集合：** <https://huggingface.co/collections/allenai/molmomotion>
- **数据集：** <https://huggingface.co/datasets/allenai/molmo-motion-1m>
- **基准：** <https://huggingface.co/datasets/allenai/PointMotionBench>
- **入库日期：** 2026-06-18
- **一句话说明：** 以 **Molmo 2 VLM 骨干** 做 **语言条件 3D 点轨迹预测**：给定 RGB 观测、物体 query 点与动作描述，预测未来数秒内各点在 **世界坐标 metric 3D** 中的轨迹；发布 **MolmoMotion-1M**（116 万视频、736 动作类）与 **PointMotionBench**（2.7K 人工校验 clip），下游用于 **机器人规划** 与 **轨迹条件视频生成**。

## 核心摘录

### 问题与动机

- **感知 vs 预测：** 现代模型能高置信 **追踪已发生** 的运动，但机器人抓取、视频生成等需要 **前瞻** 物体将如何移动。
- **任务形式：** 给定 **短视觉历史 + 物体上 3D query 点 + 语言动作描述** → 预测各点未来 **3D 轨迹**（goal-conditioned 3D point motion forecasting）。

### 运动表示：物体附着 3D 点

选用 **世界坐标系中的稀疏表面点** 作为通用运动表示，满足三点：

1. **Class-agnostic** — 不绑定人体/手/刚体模板。
2. **View-stable** — 同一物理运动跨相机/视角一致。
3. **可直接下游使用** — 紧凑显式 3D 轨迹可喂给策略或视频模型。

训练时每物体 **8 个 query 点**（足够轨迹、不足以稠密重建表面；复杂可变形运动仍受限）。

### 模型架构（Molmo 2 骨干）

- **输入：** RGB 观测 token、动作描述 text token、从 Molmo 2 视觉编码器采样的 **2D query 点特征 token**、初始 3D 坐标。
- **两变体：**
  - **MolmoMotion-AR（自回归）：** 3D 坐标 **结构化文本化** 逐步预测；条件于已生成轨迹，rollout 更平滑，**确定性路径** 上精度最高。
  - **MolmoMotion-FM（flow matching）：** 在 **连续 3D 空间** 从噪声变换为轨迹，适合 **多模态/不确定** 未来。
- **公开权重示例：** `MolmoMotion-4B-H1-F32`（4B VLM，H=32 未来帧等配置见 HF）。

### 数据：MolmoMotion-1M 标注管线

互联网视频 **缺 3D 标注** → 自动管线：

1. 根据动作描述 **ground 运动物体** 并采样 query 点。
2. **稠密 2D 跟踪** → **lift 到共享 metric 3D 帧**。
3. **物体级时空一致性** 过滤不可靠轨迹；平滑；**裁剪到物体实际运动窗口**（过滤静止段）。

规模：**1.16M 视频**、**736  motion types**、**5.6K  distinct objects**。

### 评测：PointMotionBench

- **2.7K clip**，**111  object categories**，**61  motion types**（室内操作、ego 手–物、户外动态）。
- 输入：当前观测 + query 点 + 动作描述；指标：**3D average displacement error（米）**。
- **MolmoMotion-AR (3f)** 在 HOT3D / DAVIS 等 split 上 **显著优于** 像素视频生成器（Wan2.2、Cosmos Predict 等）、Track2Act、WorldTrack 与 **常速/静态基线**。

### 下游 1：机器人规划（DROID 微调）

- 人手抬杯 vs 夹爪抬杯 **动作不同、杯子 3D 路径相似** → 3D 运动先验可 **跨 embodiment 迁移**。
- 在 **DROID** 上微调后，**MolmoBot** 策略（flow-matching action head，20K episodes）：
  - 仿真 pick-and-place **闭环成功率 76.3%** vs Molmo 2 初始化 **56.0%**。
  - **10K steps 达 51%** vs Molmo 2 版 **19%**；真机 **~2K steps** 达到 Molmo 2 **12K steps** 的 test L2。

### 下游 2：轨迹条件视频生成（DaS + MolmoMotion）

- 将 MolmoMotion 预测路径 **注入 image-to-video**（如 CogVideoX-5B），相对纯文本 prompt **更贴合小幅度精确运动**。
- **DaS + MolmoMotion** 在五项 motion 相关指标上 **全面优于** CogVideoX-5B，并在 **4/5 项** 上优于更大 **Wan2.2-I2V-A14B**。

### 局限（博客）

- 每物体 **8 点** 限制对 **复杂可变形** 运动的表面几何表达。
- 仍依赖 **深度/跟踪管线质量** 与 **语言–物体 grounding** 正确性。

## 对 wiki 的映射

- [MolmoMotion](../../wiki/entities/molmo-motion.md)
- [Generative World Models](../../wiki/methods/generative-world-models.md)
- [Manipulation](../../wiki/tasks/manipulation.md)
- [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)
