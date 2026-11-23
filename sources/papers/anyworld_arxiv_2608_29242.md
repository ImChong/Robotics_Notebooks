# anyworld_arxiv_2608_29242

> 来源归档（ingest · arXiv:2608.29242v2 全文核对）

- **标题：** AnyWorld: Factorized Egocentric World Models for Cross-Embodiment Generalization
- **短名：** AnyWorld
- **类型：** paper
- **来源：** arXiv preprint
- **原始链接：**
  - <https://arxiv.org/abs/2608.29242>（v2，2026-09-01）
  - <https://arxiv.org/html/2608.29242>
- **项目页：** <https://xpeng-robotics.github.io/anyworld/> — [`sources/sites/xpeng-robotics-anyworld.md`](../sites/xpeng-robotics-anyworld.md)
- **代码仓：** <https://github.com/xpeng-robotics/AnyWorld>（Apache-2.0，2026-09-02 发布）— [`sources/repos/anyworld.md`](../repos/anyworld.md)
- **作者：** Cheng Chen, Jerry Bai († Project Leader), Jiacheng Wei, Boyu Chen, Xiaoji Zheng, Fan Wu, Minghao Yang, Tianrun Chen, Ruibo Li, Xiaoyu Yue, Xiaoyang Guo, Yixiao Ge, Guosheng Lin, Fayao Liu (‡ Corresponding Author)
- **机构：** 南洋理工大学（NTU）；A*STAR 先进智能与计算研究所（IAIC）；小鹏机器人（XPENG Robotics）；浙江大学（ZJU）；香港中文大学（CUHK）
- **版本：** arXiv:2608.29242v2（2026-09-01）
- **入库日期：** 2026-09-02
- **核对范围：** 通读 `arXiv:2608.29242v2` HTML 全文（§1–6、Table 1–5、Fig. 1–6）与项目页 <https://xpeng-robotics.github.io/anyworld/>，并核实官方源码仓 `xpeng-robotics/AnyWorld@2ed9d9c`。未在 GPU 上运行 14B 扩散模型推理。
- **通读：** 已完成
- **一句话说明：** 将 egocentric 交互分解为动作、相机与具身三因子，无需人–机配对演示即可把单条人类交互重组合成多具身机器人域 rollout，并作为 VLA 适配阶段的数据引擎。

## 核心摘录（编译自 arXiv:2608.29242v2 / 项目页）

1. **问题：** 接触丰富机器人经验难规模化；人类 egocentric 视频丰富但每条只对应单一身体/视角/场景。目标是把单次录制当作可重组种子，生成多样机器人域经验。
2. **因子分解（§3.2）：** **动作** $A_{1:T}$ — 渲染骨架控制视频（像素平面、具身无关，编码为 $Z^a_{1:T}$）；**相机** $C_{1:T}$ — 内外参转为 Plücker 射线嵌入 $Z^c_{1:T}$；**具身** — 编辑首帧 $\tilde I^e_0$ + 文本标签 $\tau_e$（human / RoboCasa GR1 / IRON）。
3. **骨干与条件注入（§3.3）：** latent diffusion video model（DiT），初始化自 **WAN Fun-Control 14B**；动作 latent 与噪声 latent 通道拼接（$H_{\mathrm{in}}=\mathrm{Concat}[Z_s, Z^a_{1:T}, Z^0_{1:T}]$）；相机经轻量 adapter 加到 patch embedding；具身/字幕经 cross-attention 注入各 DiT block。
4. **训练（§3.4, §4.1, §4.4）：** Stage 1 — EgoDex 200K clip、30K step 预训练；Stage 2 — 无配对混合具身微调 5K step（EgoDex 5K + 目标域 GR1/IRON 各 5K clip；人:机 **2:1** 平均可控性最优：Avg 0.740 vs 1:1 0.723 vs 4:1 0.706）。全程 **无** clip 级人–机配对。
5. **可控性与视频质量（§4.3, Table 1–2）：** 60 视频评测 ActionAlign **0.659** / CameraAlign **0.789** / EmbodAcc **0.886** / Avg **0.778**（WAN Fun-Control 0.609，Cosmos-Predict2.5 0.417）；VBench 四项均分 **0.971**（Subject 0.942 / Background 0.949 / Flicker 0.995 / Smoothness 0.996）。
6. **VLA 下游与定向干预（§4.5–4.6, Table 4–5）：** 基线为已用 EgoDex 预训练的 **UniT**；适配阶段 1:1 混入 AnyWorld 重组 rollout。RoboCasa GR1 18 项 pick-and-place：**49.8%→54.6%**（+4.8 pp）；真机 IRON 20 次抓香蕉：**20.0%→55.0%**（+35.0 pp）。定向干预：（a）部分完成态「假完成先验」— 重场景 + 标定动作恢复放置；（b）语言空间目标 — 仅动作反事实不稳定，需 **联合视觉–动作重组**。
7. **局限（§6）：** 无触觉/力反馈；依赖可靠动作–相机提取（遮挡、快速运动或剧烈相机抖动会降质）；当前仅覆盖 3 具身。
8. **开源（2026-09-02 发布）：** 官方发布 [xpeng-robotics/AnyWorld](https://github.com/xpeng-robotics/AnyWorld)（Apache-2.0），包含基于 DiffSynth-Studio / Qwen-Image-Edit-2511 的 `image_editing/` 与基于 FastVideo / Wan2.1 Combined-Control 的 `world_model/` 推理代码；模型权重需另行准备基座与适配权重。

## 对 wiki 的映射

- 升格 [AnyWorld 论文实体](../../wiki/entities/paper-anyworld.md)
- 源码仓归档：[`sources/repos/anyworld.md`](../repos/anyworld.md)
- 交叉 [UniT](../../wiki/entities/paper-unit-unified-physical-language.md)（VLA 基线）、[DreamDojo](../../wiki/entities/paper-hrl-stack-35-dreamdojo.md)（人视频预训练 WM）
- 概念 [World Action Models](../../wiki/concepts/world-action-models.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)
