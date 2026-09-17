# SCAIL-2 — 项目页

- **来源：** <https://teal024.github.io/SCAIL-2/>
- **类型：** site（论文项目页）
- **机构：** 清华大学 · Z.ai（智谱）
- **归档日期：** 2026-09-17
- **论文：** <https://arxiv.org/abs/2606.10804>
- **代码：** <https://github.com/zai-org/SCAIL-2>（见 [`sources/repos/scail-2.md`](../repos/scail-2.md)）
- **权重：** <https://huggingface.co/zai-org/SCAIL-2>
- **ModelScope：** <https://modelscope.cn/models/ZhipuAI/SCAIL-2>

## 一句话说明

SCAIL-2 用 **端到端 latent 视频扩散** 统一受控角色动画子任务（单/多角色动画、跨身份替换等），以 **In-Context Mask Conditioning + Mode-Specific RoPE** 替代骨架/背景 inpainting 等中间表示，并用 **MotionPair-60K** 合成数据与 **Bias-Aware DPO** 提升细粒度区域保真。

## 为什么值得保留

- 官方演示覆盖单角色、多角色、替换与零样本跨物种/跨具身驱动，是「**去中间表示的 character animation**」主线锚点。
- 页眉链到 arXiv、GitHub、Hugging Face，便于步骤 2.5 开源核查。
- 与 SCAIL-1（arXiv:2512.05905）形成同组演进叙事。

## 开源状态（2026-09-17 核查）

| 项 | 结论 |
|----|------|
| 项目页 Code / HF 链接 | **已列** GitHub + Hugging Face |
| 推理代码 | **已开源** — `zai-org/SCAIL-2` 主分支 `generate.py` |
| 训练代码 | **已开源** — `sat-scail2` 分支（2026-08-06 发布） |
| 权重 | **已发布** — HF `zai-org/SCAIL-2`（MIT）；含 DPO LoRA、Relighting LoRA |
| 预处理 | **已开源** — 子模块 `SCAIL-Pose`（NLF + DWPose + SAM3 掩码管线） |
| 数据集 MotionPair-60K | **合成数据** — 论文描述由 SCAIL-Preview / Wan-Animate / MoCha 管线生成；无独立 HF dataset 链接 |
| 结论 | **已开源**（推理 + 训练 + 权重 + 预处理）；数据集以论文/合成管线描述为主 |

## 对 wiki 的映射

1. **[SCAIL-2（论文实体页）](../../wiki/entities/paper-scail-2.md)**
2. **[sources/papers/scail2_arxiv_2606_10804.md](../papers/scail2_arxiv_2606_10804.md)**
3. **[Character Animation vs Robotics](../../wiki/concepts/character-animation-vs-robotics.md)** — 纯生成式角色动画 vs 物理可控机器人边界
