# perceptron_isaac_05

> 来源归档（technical report）

- **标题：** Isaac 0.5: Percepts Scale Control / Isaac 0.5: An Open-Weight Embodied Foundation Model
- **类型：** paper（公司技术报告，无 arXiv ID）
- **机构：** 感知器（Perceptron）
- **PDF：** https://pub-d90b81cad7254a1aa6b148ac18153c0c.r2.dev/isaac-0.5.pdf
- **配套博客：** https://www.perceptron.inc/blog/introducing-isaac-0-5
- **代码：** https://github.com/perceptron-ai-inc/isaac
- **权重页：** https://huggingface.co/PerceptronAI/Isaac-0.5
- **入库日期：** 2026-08-27
- **页数：** 52（PDF 元数据日期 2026-08-26）
- **一句话说明：** Perceptron 的 36B-A2.5B 开源权重具身基础模型技术报告：null-expert 稀疏骨干 + FAST/Flow 双动作接口 + 专有未来 percept 自监督；给出无动作标签视频与遥操作的 **210.3×** 置换律，并报告感知/具身推理与 LIBERO 适配数字。
- **沉淀到 wiki：** [Perceptron Isaac 0.5](../../wiki/entities/perceptron-isaac-05.md)

## 开源状态（与博客同步，2026-08-27）

**部分开源。** 报告宣称释放权重、推理与适配代码、评测设定与 35+ 本体部署配置。GitHub 代码 **Apache 2.0** 已公开；Hugging Face 权重页截至入库日仍标 **COMING SOON**；未来 percept 的目标构造与损失实现 **专有**；mHarmony/TensorStream 运行时未钉进 LeRobot extra。详见 [perceptron_isaac_05.md](../blogs/perceptron_isaac_05.md)。

## 核心摘录

### 1) 架构（§3）

- **共享骨干：** Qwen-family VLM；patch 16；宽 2048 的 **40** 稀疏块（**30 GDN** 线性注意力 + **10** 全注意力）。
- **Null-expert MoE：** 每层 256 真实 routed MLP + 1 个学习到的 null 行（复制为 256 个 null 候选）+ 常开 shared expert；top-8；每 token 真实专家数 **0–8**。
- **规模：** checkpoint **36B** 总参数；参考设定下期望 **2.5B** 激活（8 路中 4 路为真实）；激活范围约 **0.1B–3B**。
- **离散控制：** 与语言/坐标共用自回归接口，另设 **2,048** 词表的 FAST 动作 token（Pertsch et al., 2025）。
- **连续控制：** 骨干状态 2048→768 作 DiT 交叉注意力 K/V；**36 块 DiT**（宽 768、8 头）+ Flow expert；训练每 chunk **S=4** 独立噪声与 Flow-time 样本。

### 2) 数据混合物（§2）

- **529** 源流水；调度质量：感知/具身推理 **69.7%** vs 机器人 **30.3%**；打包后 token 暴露 **反转** 为 **20.4% / 79.6%**（机器人样本膨胀更多 packed 位置）。
- 感知侧 **3T** native token。机器人侧 **100,000 h / 35+** 本体配置：高质量 teleop **10,000 h**、更广交互 **40,000 h**、游戏与仿真 **50,000 h**。
- 视频缩放实验网格：通用无动作视频 **1,000,000 h**、egocentric **375,000 h**、UMI **375,000 h**（固定 80:30:30）。

### 3) Scaling law（§4）

- 固定视频组成，相对 teleop 扫网格；主阈值 held-out action loss **τ = 2.50**。
- 通用视频 1,000 → 1,000,000 h：teleop 交叉点 **5,884 → 28 h**（**210.3×**；相邻 rung **83–300×**）。
- 斜率：1 h teleop 时 10× 视频几乎平坦（约 **−0.006**）；约 100 h teleop 起稳定在约 **−0.21 / 10× 视频**。
- 发布 checkpoint 对应网格最高视频预算（1M h 通用视频）。

### 4) 评测（§5）

- Grounding（Table 4，同 harness）：Isaac **CARPK 19.07 / LVIS Count 32.78 / ScreenSpot-Pro 62.60**。
- LIBERO 按套件适配（每套 10 任务、**500** 演示；Table 10）：Spatial **98.0** / Object **99.0** / Goal **98.8** / Long **93.0**，平均 **97.2**（与 MolmoAct2 **97.2**、GR00T N1.7 **97.0**、π0.5 **96.9** 同档；协议来自各原文，**不可直接当统一重跑**）。
- 部署足迹（Table 11，架构估算）：H100 SXM 上三张 1024² 图 + 10 Flow 步，延迟 **70 ms**、策略率 **14.3 Hz**、FP8 checkpoint **36.0 GB**。
- 国际象棋物理操作：同一 checkpoint 读棋盘并出控；one-epoch / 单 expert episode 适配增益相对 SmolVLA / GR00T N1.7 / MolmoAct2 / π0.5 最大（博客 **10.5× / 9.5× / 7.0×**）。

### 5) 系统（§3.6 / §6）

- **mHarmony** typed 事件编译 → **TensorStream** packed 序列。
- 百万小时视频训练栈：预测式 shard 流、节点共享下载、进程隔离解码、best-fit packing；hypersparse 设定 MFU **24%**（2.5B/36B 激活）。

## 对 wiki 的映射

- 实体：[wiki/entities/perceptron-isaac-05.md](../../wiki/entities/perceptron-isaac-05.md)
- 方法/概念：[wiki/methods/vla.md](../../wiki/methods/vla.md)、[wiki/concepts/foundation-policy.md](../../wiki/concepts/foundation-policy.md)、[wiki/concepts/embodied-scaling-laws.md](../../wiki/concepts/embodied-scaling-laws.md)
- 工程：[wiki/entities/lerobot.md](../../wiki/entities/lerobot.md)
- 消歧：[wiki/entities/isaac-gr00t.md](../../wiki/entities/isaac-gr00t.md)

## 当前提炼状态

- [x] 技术报告架构 / 数据 / scaling / 评测数字摘录
- [x] 与博客、GitHub README、Hub 页交叉核对开源边界
- [ ] 若后续上 arXiv，补 `arxiv:` 字段并改文件名约定
