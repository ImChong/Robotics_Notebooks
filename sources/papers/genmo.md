# GENMO（A GENeralist Model for Human MOtion）

> 来源归档（ingest）

- **标题：** GENMO: A GENeralist Model for Human MOtion
- **类型：** paper
- **来源：** arXiv abs / arXiv HTML v1
- **原始链接：**
  - <https://arxiv.org/abs/2505.01425v1>
  - <https://arxiv.org/html/2505.01425v1>
  - 项目页（重定向至 GEM）：<https://research.nvidia.com/labs/dair/gem/>
  - 代码：<https://github.com/NVlabs/GENMO>
- **作者：** Jiefeng Li, Jinkun Cao, Haotian Zhang, Davis Rempe, Jan Kautz, Umar Iqbal, Ye Yuan
- **机构：** NVIDIA Research（DAIR 等）
- **会议：** ICCV 2025（Highlight）；项目后续以 **GEM** 名义发布权重与扩展
- **入库日期：** 2026-05-16
- **最后更新：** 2026-05-16
- **一句话说明：** 把人体运动 **估计** 与 **生成** 统一进同一个扩散模型：将估计形式化为「带观测约束的生成」，再以 **dual-mode 训练范式** 让首步预测足够准确，从而用同一套权重覆盖视频/2D 关键点/文本/音乐/3D 关键帧条件下的全身 SMPL 序列恢复与合成。

## 核心论文摘录（MVP）

### 1) 问题动机：估计 vs 生成的人为分裂

- **链接：** <https://arxiv.org/abs/2505.01425v1>
- **摘录要点：** 现有人体运动建模长期将「生成」（diverse / plausible，从文本/音频/关键帧出发）与「估计」（accurate / 唯一，从视频/2D 等观测出发）分作两类模型；二者共享时间动力学与运动学表示，但分离的范式既阻断知识迁移，又要维护两套权重。论文主张：可以把估计**重新表述为生成的一种**——「在观测条件约束下的运动生成」——从而在单一框架里同时获得生成的先验弹性与估计的精度。
- **对 wiki 的映射：**
  - [GENMO（统一人体运动估计与生成）](../../wiki/methods/genmo.md) — 作为方法页第一段「为什么重要」的论据。
  - [Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md) — 将 GENMO 列为「人体运动域」典型实例，与机器人控制域的扩散生成相互参照。

### 2) 方法：Unified Estimation & Generation 架构

- **链接：** <https://arxiv.org/html/2505.01425v1>
- **摘录要点：**
  - **联合表示**：每帧 $x^i = (\Gamma_{\text{gv}}^i, v_{\text{root}}^i, \theta^i, \beta^i, t_{\text{root}}^i, \pi^i, p^i)$，同时编码 gravity-view 全局轨迹、egocentric 局部 SMPL 参数（24×6 关节角 + 10 维 shape + root 平移）、相机外参 $\pi^i$ 与手脚接触标签 $p^i$。这套 **gravity-view + SMPL + camera + contact** 的拼接，让单一序列同时承载估计任务（要求相机一致）与生成任务（要求 heading-free）。
  - **条件集合** $\mathcal{C}$：视频特征、相机运动、2D 骨架、音乐片段、2D bbox、自然语言文本；每种条件有同尺寸的 mask $m_\star$ 表示哪些帧 / 哪些 token 可用。
  - **加性融合**：所有 frame-aligned 条件经各自 MLP 编码后**逐 token 相加**得到统一条件 token；再与噪声运动 $x_t$ 融合进入主干。
  - **RoPE Transformer + Multi-Text Injection**：主干由 $L$ 个「RoPE 块 + multi-text 注入块」级联。相对位置编码使得序列长度可在推理期任意扩展；推理时用 W 帧滑窗注意力既支持「比训练长得多」的序列，又控制算力。
  - **多文本注意力**：对 $K$ 段文本 $c_{\text{text}}^k$，每段绑定一个时间窗口 mask $\Omega_k$，文本特征只对落在窗内的运动 token 起作用；通过 $\sum_k \text{MaskedMHA}(f_{\text{in}}, c_{\text{text}}^k, \Omega_k)$ 实现「同一序列里多段语义指令叠加」的可编辑生成。
  - **混合多模态条件**：text 走 multi-text 通路；frame-aligned 模态（视频/音乐/2D 骨架）通过 mask 与时间窗的乘法门控自由插拔，使用户可以「前半段用视频驱动，中段用文本，后段切回音乐」。
- **对 wiki 的映射：**
  - [GENMO（统一人体运动估计与生成）](../../wiki/methods/genmo.md) — 作为方法页「主要技术路线」表格与流程图骨架。
  - [扩散运动生成方法对照阅读](../../wiki/methods/diffusion-motion-generation.md) — 多文本/多模态条件耦合的具体范例。

### 3) Dual-Mode 训练范式与 estimation-guided generation

- **链接：** <https://arxiv.org/html/2505.01425v1#S3.SS2>
- **摘录要点：**
  - **观察**：视频条件下扩散预测**几乎确定**（第一步去噪结果与最终结果差异极小），而文本条件下方差很大。这意味着估计任务的「首步预测」必须够好，否则后续 DDIM 步无法把误差拉回。
  - **估计模式**：输入纯高斯噪声 $z\sim\mathcal{N}(0,I)$ 与**最大时间步 $T$**，以 MSE 学习 $\mathcal{L}_{\text{est}} = \mathbb{E}_z[\|x_0 - \mathcal{G}(z, T, \mathcal{C}, \mathcal{M})\|^2]$，等价于做条件回归——但**与扩散噪声日程兼容**，从而与生成模式共存。再叠加 **geometric loss $\mathcal{L}_{\text{geo}}$**（关节/顶点的相机/世界坐标位置 + 接触约束）与 2D 重投影损失。
  - **生成模式**：清洁 3D 数据走标准 DDPM 目标 $\mathcal{L}_{\text{gen}}$；只有 2D 标注的野外视频走 **estimation-guided generation**：先用估计模式得到伪 clean 运动 $\hat{x}_0$，再前向加噪 $\hat{x}_t$，用 2D 重投影损失 $\mathcal{L}_{\text{gen-2D}} = \mathbb{E}[\|x_{\text{2d}} - \Pi(\mathcal{G}(\hat{x}_t, t, \mathcal{C}))\|^2]$ 训练。如此**2D 弱监督也能拉宽生成分布**，而不必显式重建 3D ground-truth。
  - **模式选择规则**：强条件数据（视频、2D 骨架）同时用估计 + 生成两模式；弱条件数据（纯文本、纯音乐）只用生成模式。两种模式都加 $\mathcal{L}_{\text{geo}}$。
- **对 wiki 的映射：**
  - [GENMO（统一人体运动估计与生成）](../../wiki/methods/genmo.md) — 沉淀「双模式训练 + estimation-guided 2D 监督」的主表与流程图。
  - [Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md) — 与「闭环微调修正分布偏移」对照：GENMO 通过 dual-mode 在训练阶段解决「首步必须准」的同类问题。

### 4) 实验结论（公开 v1 表述）

- **链接：** <https://arxiv.org/html/2505.01425v1#S4>
- **摘录要点：**
  - **全局动作估计**：在 EMDB-2、RICH 上的 WA-MPJPE100 / W-MPJPE100 / RTE / Jitter / Foot-Sliding 等指标上对比 GLAMR、TRAM、WHAM 等基线；论文宣称在全局动作估计任务上达到 SOTA。
  - **局部动作估计 + 音乐到舞蹈生成**：同一权重在局部估计 (3DPW 等) 与 music-to-dance 等生成任务上均达到或接近 SOTA。
  - **关键消融**：(a) 去掉估计模式 → 估计精度显著下降；(b) 去掉 estimation-guided 2D 生成 → 生成多样性下降；(c) 去掉 multi-text 注意力 → 多段文本条件下出现时间错位。
  - **遮挡 / 截断鲁棒性**：生成先验显式提升困难帧上的估计质量，体现「生成 ↔ 估计」的双向收益。
- **对 wiki 的映射：**
  - [GENMO（统一人体运动估计与生成）](../../wiki/methods/genmo.md) — 在「常见误区或局限」与「关联页面」部分使用，强调「同模型多任务」叙事的强支撑与实验前提。

### 5) 与下游人形控制 / 视频驱动管线的耦合

- **链接：** <https://github.com/NVlabs/GENMO>、<https://research.nvidia.com/labs/dair/gem/>
- **摘录要点：** README 把 GEM/GENMO 定位为「NVIDIA 人形机器人栈」中的人体运动接口环节，关联工作包括 [GEM-X](https://github.com/NVlabs/GEM-X)（全身 + 手脸）、[SOMA-X](https://github.com/NVlabs/SOMA-X) body model、BONES-SEED 数据集、[ProtoMotions](https://github.com/NVlabs/ProtoMotions)、SOMA Retargeter、[SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) 与 [Kimodo](https://github.com/nv-tlabs/kimodo)。在 [ExoActor](https://baai-agents.github.io/ExoActor/) 中，GENMO 被作为「生成视频 → SMPL 全身轨迹」的标准中间件，与 WiLoR（手部）+ SONIC（跟踪控制器）拼成端到端的 video-to-G1 管线。
- **对 wiki 的映射：**
  - [ExoActor (视频生成驱动的交互式人形控制)](../../wiki/methods/exoactor.md) — 已经把 GENMO 列为系统集成位之一；本次 ingest 在 GENMO 方法页补全这一双向引用。
  - [SONIC（规模化运动跟踪人形控制）](../../wiki/methods/sonic-motion-tracking.md) — 在「公开材料要点」中明确「视频/文本/音乐 → GEM → SONIC」的官方演示路径，GENMO 方法页应反向链回。
  - [ProtoMotions](../../wiki/entities/protomotions.md) — 作为同一 NVIDIA 人形栈中的训练框架，间接利用 GEM/GENMO 输出的人体运动作训练参考。

## 当前提炼状态

- [x] arXiv abs / HTML v1 的摘要、方法节（§3.1 架构、§3.2 dual-mode）、实验表头与与下游栈的关联已摘录
- [x] 与 `sources/repos/genmo.md`（聚焦代码仓与 GEM 重命名）分工明确，本文件聚焦论文方法
- [x] wiki 映射：`wiki/methods/genmo.md` 本次同步补充「双模式训练 / 多文本注入 / NVIDIA 人形栈」三块，并新增 Mermaid 流程图
- [ ] 若后续 v2 / 评测细节 / GEM-X 论文公开，可在此追加「实验细节」与「全身扩展」摘录段
