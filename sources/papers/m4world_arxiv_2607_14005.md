# M⁴World: A Multi-view Multimodal Driving World Model for Interactive Object Manipulation and Minute-long Streaming（arXiv:2607.14005）

> 来源归档（ingest）

- **标题：** M⁴World: A Multi-view Multimodal Driving World Model for Interactive Object Manipulation and Minute-long Streaming
- **缩写：** **M⁴World** / **M4World**（文中写作 M${}^{\text{4}}$World；四重 M：Multi-view、Multimodal、Manipulation、Minute-long）
- **类型：** paper / generative driving world model / multi-view video + LiDAR / object-level controllability
- **arXiv：** <https://arxiv.org/abs/2607.14005>（PDF：<https://arxiv.org/pdf/2607.14005.pdf>；HTML：<https://arxiv.org/html/2607.14005>）
- **项目页 / 代码：** 截至入库日 **未发现** 官方项目页或 GitHub；arXiv 条目仅论文本体（无 Code / Project 链接）
- **作者：** Ke Cheng*、Hanqiao Ye*、Lei Shi、Yahui Liu、Yunhan Shen、Jingtao Dong、Zhenke Wang、Wenxuan Ao、Weixiang Xu、Kaining Huang、Shuhan Shen†（*Equal contribution；†Corresponding author）
- **机构：** 美团（Meituan）；中国科学院自动化研究所（CASIA）；北京理工大学（BIT）
- **入库日期：** 2026-07-22
- **一句话说明：** **多视角多模态** 驾驶世界模型：在共享 DiT 潜空间上联合生成 **环视视频 + 同步 LiDAR range map**，以融合几何/外观/文本的 **物体 token** 做可交互对象操纵，并经多阶段因果蒸馏实现 **分钟级** 流式 rollout（4 步去噪）。

## 开源状态（arXiv 核查，2026-07-22）

- **未开源：** arXiv abs / HTML / PDF **均未列出** 项目页或代码仓库；论文正文亦无「code will be released」类可核验外链。训练数据为自采集驾驶日志，不可公开复现完整管线。
- **可复用先验：** 骨干初始化自开源 **Wan2.1-T2V**；对比基线为 **MagicDriveV2**；评测引入 **VLM judge**（Qwen3-VL 等）协议，可作方法对照而非权重复现。

## 摘要级要点

- **缺口：** 现有驾驶生成 WM 多停留在 **几何条件**（3D box / occupancy），外观级对象控制弱；长视界因果流式易暴露 bias 或延迟高。
- **架构：** 共享 **DiT** + 双交叉注意力（控制信号 / 跨视角聚合）；控制集含场景文本、相机外参、自车位姿、BEV map、**物体 token**（8 角点 box + 类别 + SigLIP-V2 图像描述 + umT5 文本描述）。
- **LiDAR：** 128-beam 扫描投影为 **range map**，与相机共享 video VAE 潜空间，并行多模态生成；采样用 **APG** 抑制 CFG 导致的深度偏移。
- **训练五阶段：** 双向 mid-training → Teacher Forcing 因果 → 4-step ODE 学生初始化 → Self-Forcing + 非对称 DMD → 长视频迭代微调；推理加 **latent context refresh** 缓解 chunk 边界闪烁。
- **长尾定制：** 每稀有 case **LoRA + 50/50 稀有/普通采样** few-clip 后训练；另提供首帧多/单视角条件与物体补全等 **视觉参考条件** 变体。
- **数据：** ~4 万条 10s 短 clip + ~4k 条 60s 长 clip；10 相机 @10FPS + 1×128-beam LiDAR；自动打标场景/物体描述。
- **结果索引：** vs MagicDriveV2：FID/FVD **41.7/346.1 → 34.8/288.7**；物体 visual/textual fidelity **13.4%/11.6% → 62.7%/59.1%**；跨视角一致 **78.9% → 84.5%**；8×A100 上 6 摄+LiDAR：**2.3 FPS**（424×800）；运树卡车长尾：50k 真实 + 500 合成 → recall **1.0% → 69.7%**，常规 mAP 基本不变。

## 核心论文摘录（MVP）

### 1) 问题形式：因果 chunk 潜空间世界模型

- **链接：** §3.1；Eq. (1)；Fig. 1
- **摘录要点：** 在 video VAE 潜空间自回归采样下一 chunk $\hat{\mathbf{z}}_{T+1:T+W}\sim p_\theta(\cdot\mid\mathbf{z}_{1:T},\mathbf{C}_{T+1:T+W})$，解码为时间对齐的环视视频与 LiDAR range map。由开源双向 T2V（Wan2.1）经多阶段改编为可控多模态驾驶 WM。
- **对 wiki 的映射：**
  - [M⁴World](../../wiki/entities/paper-m4world.md) — 主实体与流程图。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 驾驶多模态分支。

### 2) 物体级控制 token：几何 + 外观

- **链接：** §3.2；Object tokens
- **摘录要点：** 相对 MagicDrive / MagicDriveV2 等几何条件，每个物体 token 融合 3D box、类别、SigLIP-V2 图像外观与 umT5 细粒度文本，使仿真可指定「放在哪」与「长什么样」——对长尾安全关键场景构造至关重要。
- **对 wiki 的映射：**
  - [M⁴World](../../wiki/entities/paper-m4world.md) — 控制接口表。
  - [X-World](../../wiki/entities/paper-x-world.md) — 同为多摄驾驶 WM，但条件侧重动作/外观文本而非物体图像描述。

### 3) 长视界流式：因果蒸馏 + context refresh

- **链接：** §4；Fig. 11
- **摘录要点：** Teacher Forcing → 4-step ODE 学生 → Self-Forcing + asymmetric DMD（critic 5 次更新 / 1 次 generator；10% 混入监督去噪损失）→ 600 帧长视频迭代微调。推理时将上 chunk 末帧 latent 作下一 chunk 显式视觉锚点（keys/values 需重算，非纯 KV 滚动）。
- **对 wiki 的映射：**
  - [M⁴World](../../wiki/entities/paper-m4world.md) — 训练配方。
  - [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) — 分钟级像素仿真稳定性。

### 4) VLM 可控性评测与长尾增强

- **链接：** §5、§8、§9；Tab. 2–4
- **摘录要点：** 标准 FID/FVD 之外，用 VLM 二元问答评场景天气/时段、视角内物体存在/清晰/视觉保真/文本保真、跨视角同一物体一致性。运树卡车 case：few-clip LoRA 后合成 500 clip，显著抬升目标 recall。首帧编辑 + 视觉参考条件可零样本替换长尾外观并跨视角传播。
- **对 wiki 的映射：**
  - [M⁴World](../../wiki/entities/paper-m4world.md) — 评测与下游。
  - [自动驾驶核心算法盘点](../../wiki/overview/autonomous-driving-core-algorithms-series.md) — 感知增广坐标。

## 对 wiki 的映射（汇总）

- 主实体页：[`wiki/entities/paper-m4world.md`](../../wiki/entities/paper-m4world.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[X-World](../../wiki/entities/paper-x-world.md)、[PanoWorld](../../wiki/entities/paper-panoworld-real-world-panoramic-generation.md)、[robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)、[自动驾驶核心算法盘点](../../wiki/overview/autonomous-driving-core-algorithms-series.md)
