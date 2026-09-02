# anyworld_arxiv_2608_29242

> 来源归档（ingest）

- **标题：** AnyWorld: Factorized Egocentric World Models for Cross-Embodiment Generalization
- **短名：** AnyWorld
- **类型：** paper
- **来源：** arXiv preprint
- **原始链接：**
  - <https://arxiv.org/abs/2608.29242>
  - <https://arxiv.org/html/2608.29242>
- **项目页：** <https://xpeng-robotics.github.io/anyworld/> — [`sources/sites/xpeng-robotics-anyworld.md`](../sites/xpeng-robotics-anyworld.md)
- **作者：** Cheng Chen, Jerry Bai, Jiacheng Wei, Boyu Chen, Xiaoji Zheng, Fan Wu, Minghao Yang, Tianrun Chen, Ruibo Li, Xiaoyu Yue, Xiaoyang Guo, Yixiao Ge, Guosheng Lin, Fayao Liu
- **机构：** 南洋理工大学（NTU）；A*STAR 先进智能与计算研究所（IAIC）；小鹏机器人（XPENG Robotics）；浙江大学（ZJU）；香港中文大学（CUHK）
- **版本：** arXiv:2608.29242（2026）
- **入库日期：** 2026-09-02
- **一句话说明：** 将 egocentric 交互分解为动作、相机与具身三因子，无需人–机配对演示即可把单条人类交互重组合成多具身机器人域 rollout，并作为 VLA 适配阶段的数据引擎。

## 核心摘录（编译自 arXiv / 项目页，细节以原文为准）

1. **问题：** 接触丰富机器人经验难规模化；人类 egocentric 视频丰富但每条只对应单一身体/视角/场景。目标是把单次录制当作可重组种子，生成多样机器人域经验。
2. **因子分解：** **动作** — 渲染骨架控制视频（像素平面、具身无关）；**相机** — Plücker 射线嵌入；**具身** — 首帧 + 文本标签 \(\tau_e\)（human / RoboCasa GR1 / IRON）。
3. **骨干：**  latent diffusion video model（DiT）；动作 latent 通道拼接；相机经轻量 adapter 加到 patch embedding；具身/字幕经 cross-attention 注入。初始化自 **WAN Fun-Control 14B**。
4. **训练：** Stage 1 — EgoDex 20 万 clip、30K step 预训练；Stage 2 — 无配对混合具身微调 5K step（EgoDex + RoboCasa GR1 + IRON，各域 5K clip；人:机 **2:1** 最优）。全程 **无** clip 级人–机配对。
5. **可控性（60 视频）：** ActionAlign **0.659** / CameraAlign **0.789** / EmbodAcc **0.886** / Avg **0.778**（优于 Cosmos-Predict2.5 与 WAN Fun-Control）；VBench 四项均 **0.971**。
6. **VLA 下游：** 基线为已用 EgoDex 预训练的 **UniT**；适配阶段 1:1 混入 AnyWorld 重组 rollout。RoboCasa GR1 18 项 pick-and-place：**49.8%→54.6%**；真机 IRON 20 次抓香蕉：**20.0%→55.0%**。
7. **定向干预：** （a）部分完成态「假完成先验」— 重场景 + 标定动作可恢复任务；（b）语言空间目标 — 仅动作反事实配对 **不能** 稳定指令跟随，需 **联合视觉–动作重组**。
8. **局限：** 无触觉/力反馈；依赖可靠动作–相机提取；当前仅 3 具身；更长程与更多形态待扩展。
9. **开源：** 截至 2026-09-02，项目页与 `xpeng-robotics` GitHub 组织 **未列** 训练/推理/权重仓库。

## 对 wiki 的映射

- 升格 [AnyWorld 论文实体](../../wiki/entities/paper-anyworld.md)
- 交叉 [UniT](../../wiki/entities/paper-unit-unified-physical-language.md)（VLA 基线）
- 交叉 [DreamDojo](../../wiki/entities/paper-hrl-stack-35-dreamdojo.md)（人视频预训练 WM）
- 概念 [World Action Models](../../wiki/concepts/world-action-models.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)
