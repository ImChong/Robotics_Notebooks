# DriftWorld: Fast World Modeling through Drifting（arXiv:2607.15065）

> 来源归档（ingest）

- **标题：** DriftWorld: Fast World Modeling through Drifting
- **类型：** paper / action-conditioned video world model / drifting generative models / policy evaluation
- **arXiv：** <https://arxiv.org/abs/2607.15065>（PDF：<https://arxiv.org/pdf/2607.15065.pdf>）
- **项目页：** <https://susie-lu.github.io/driftworld/>
- **代码：** <https://github.com/Susie-Lu/driftworld>
- **权重：** <https://huggingface.co/Susie-Lu/driftworld>
- **作者：** Susie Lu、Weirui Ye（MIT）；Haonan Chen、Yilun Du（Harvard）
- **机构：** 麻省理工学院（MIT）、哈佛大学（Harvard University）
- **入库日期：** 2026-07-22
- **一句话说明：** 将 **drifting generative models** 首次适配为 **动作条件视频世界模型**：训练期学 action-conditioned drift，推理期 **单次前向** 从当前观测 + 候选动作序列生成未来帧，H100 上 **30+ fps**（平均约 **17×** 快于扩散 WM）；支撑 **GPC-RANK 推理时动作搜索** 与 **离线策略评估**（与 GT 相关性最高约 **0.99**）。

## 开源状态（项目页 + 仓库核查，2026-07-22）

- **部分开源：** 项目页 Paper / Code 双链齐全；官方仓含 Push-T **可运行** 训练 / 可视化 / 指标 / GPC-RANK / 策略评估脚本与 HF checkpoint；README 写明其它数据集代码 **Will be added soon**；GitHub **未声明** license。

## 摘要级要点

- **瓶颈：** 扩散式动作条件 WM 多步去噪使大规模想象 rollout 昂贵（文中引用 GPC：扩散 rollout 可占决策周期 90–95%）。
- **方法：** 基于 Deng et al. *Generative Modeling via Drifting*（arXiv:2602.04770）——训练期用 attraction–repulsion drifting field 把生成分布推向数据分布，推理 **无需迭代采样**。
- **机器人适配三件套：** ① **action-accentuated** drifting（负样本混合「无动作」真实帧）；② **DINOv2/v3 特征空间** drifting（复杂实机场景保锐）；③ **帧级 FiLM 动作条件 U-Net**（每帧对应动作）。
- **基准：** Bridge-V2、RT-1、Language Table、Push-T、Robomimic（Lift / Can / Square；含双视角）。
- **结果索引：** Push-T 64-frame 视觉指标优于 GPC / AVDC / Ctrl-World / MSE baseline，timing **0.0037 s/frame**；GPC-RANK 将 Policy 1 IoU **0.635→0.781**；策略评估 Pearson **0.9916 / 0.9250 / 0.9515**（Lift / Can / Push-T）。

## 核心论文摘录（MVP）

### 1) 问题形式与 1-step drifting 生成

- **链接：** §3.1–3.2；Fig. 1
- **摘录要点：** 给定历史观测 $o_{t-F:t}$ 与未来动作 $a_{t:t+T}$，预测 $o_{t+1:t+T+1}$。生成分布为 pushforward $q=f_{\#}p_{\epsilon}$；drifting field $V_{p,q}=V_p^+-V_q^-$ 用 **单一正样本**（唯一 GT 未来）与 $N_{\mathrm{neg}}$ 负样本（模型生成）驱动固定点迭代损失。
- **对 wiki 的映射：**
  - [DriftWorld](../../wiki/entities/paper-driftworld.md) — 核心机制与流程图。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 非扩散单次前向分支。

### 2) 动作强化、特征空间与运动加权

- **链接：** §3.3；Fig. 2；Tab. 5
- **摘录要点：** 负样本混合 $\gamma$ 比例「无动作」真实帧以强化动作跟随；复杂实机数据在 **DINOv2/v3** 特征图逐空间位置算 drifting；按运动幅度 tanh 加权，避免背景 identity mapping；可选 **self-forcing** 第二阶段改善自回归。
- **对 wiki 的映射：**
  - [DriftWorld](../../wiki/entities/paper-driftworld.md) — 工程适配与消融。

### 3) 视觉质量 vs 推理速度

- **链接：** §4.2；Tab. 1–3
- **摘录要点：** 五环境上匹配或超过 IRASim / WorldGym / Ctrl-World / GPC / LVDM / VDM 等，同时大幅降低每帧时间（如 Bridge-V2 **0.0300 s** vs IRASim **1.1031 s**；项目页报 Bridge / RT-1 / Language Table 约 **33–39 fps**）。
- **对 wiki 的映射：**
  - [DriftWorld](../../wiki/entities/paper-driftworld.md) — 基准表。
  - [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) — 像素仿真速度约束。

### 4) GPC-RANK 推理时改进与离线策略评估

- **链接：** §4.3–4.4；Tab. 4；Fig. 6
- **摘录要点：** 基策略采样 $K=50$ 提案 → DriftWorld 想象 rollout → 奖励模型选最高分动作块；相对 GPC 扩散 WM 更快且 IoU 更高。离线评估：七个扩散策略 Push-T IoU 相关性 **0.9515**；Robomimic Lift/Can 成功率相关性 **0.9916 / 0.9250**。
- **对 wiki 的映射：**
  - [DriftWorld](../../wiki/entities/paper-driftworld.md) — 规划 / 评估双用途。
  - [world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md) — 虚拟沙盒外延。
  - [GigaWorld-1](../../wiki/entities/paper-gigaworld-1-policy-evaluation.md) — WM 作评估器坐标对照。

## BibTeX（官方 README）

```bibtex
@article{lu2026driftworld,
  title={DriftWorld: Fast World Modeling through Drifting},
  author={Lu, Susie and Chen, Haonan and Ye, Weirui and Du, Yilun},
  journal={arXiv preprint arXiv:2607.15065},
  year={2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-driftworld.md`](../../wiki/entities/paper-driftworld.md)
- 代码归档：[`sources/repos/driftworld.md`](../repos/driftworld.md)
- 项目页：[`sources/sites/susie-lu-driftworld-github-io.md`](../sites/susie-lu-driftworld-github-io.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md)、[OSCAR](../../wiki/entities/paper-oscar.md)、[WorldGym](../../wiki/entities/paper-shenlan-wm-15-worldgym.md)、[GigaWorld-1](../../wiki/entities/paper-gigaworld-1-policy-evaluation.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)
