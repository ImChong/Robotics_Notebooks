# Gamma-World 官方项目页（NVIDIA SIL）

> 来源归档

- **标题：** Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players — Project Page
- **类型：** site / project-page
- **URL：** <https://research.nvidia.com/labs/sil/projects/gamma-world/>
- **关联论文：** <https://arxiv.org/abs/2605.28816>
- **关联代码：** <https://github.com/nv-tlabs/Gamma-World>
- **Hugging Face：** <https://huggingface.co/papers/2605.28816>
- **机构：** NVIDIA SIL；合作方含清华、多伦多大学、Vector Institute
- **入库日期：** 2026-05-30
- **一句话说明：** **γ-World** 公开落地页：TL;DR、交互 demo 叙事、Gallery（总览 / 双人 / 四人零样本 / 真实多机协调）、Method 图示（SRAE + Sparse Hub Attention）、效率对比图与 BibTeX。

## 页面结构归纳

1. **TL;DR：** SRAE + Sparse Hub Attention → **24 FPS** rollout；**2→4 玩家零样本**；虚拟游戏到真实机器人。
2. **Gallery 分区：**
   - Overview — 多场景多配置交互生成
   - Two-Agent Interaction — 各 agent 独立可控、共享演化世界
   - Four-Agent Generalization — 置换对称编码带来的 **无额外训练** 扩展
   - Real-World Robotics Coordination — 真实多机器人协调定性
3. **Method：**
   - 输入：**per-agent action streams** → 输出：**shared multi-view rollout**
   - **Simplex Rotary Agent Encoding** — 3D RoPE 扩展，正单纯形顶点相位，置换等价
   - **Sparse Hub Attention** — hub 中介，跨 agent 注意力 **线性** 于 \(N\)
   - **Efficiency 图** — Sparse Hub vs dense cross-agent attention
4. **Citation：** `@article{gammaworld2026, ... arXiv:2605.28816}`

## 对 wiki 的映射

- 与 [`sources/papers/gamma_world_arxiv_2605_28816.md`](../papers/gamma_world_arxiv_2605_28816.md) 互为补充：论文摘录偏机制，本页偏 **演示分区与效率叙事**。
- 沉淀：[`wiki/entities/paper-gamma-world-multi-agent.md`](../../wiki/entities/paper-gamma-world-multi-agent.md)
