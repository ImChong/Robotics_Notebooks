# WM-Craftnet: World Synesthesia Model for Generalizable and Robust Dexterous In-Hand Manipulation

> 来源归档（ingest）

- **标题：** WM-Craftnet: World Synesthesia Model for Generalizable and Robust Dexterous In-Hand Manipulation
- **简称：** WM-Craftnet / WSM
- **类型：** paper / dexterous-manipulation / in-hand-rotation / world-model / visuotactile-rl / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2609.07002>
- **PDF：** <https://arxiv.org/pdf/2609.07002>
- **提交日期：** 2026-09-07
- **会议：** CoRL 2026（Austin, Texas, USA）
- **项目页：** <https://wmcraftnet.github.io/>
- **机构：** Sharpa Robotics
- **作者：** Jie Yin、Zeyuan Zhao、Xiaojing Tan、Yang Liu、Chiyu Wang、Xinyang Gu
- **平台：** Sharpa 灵巧手真机（腕部 depth + 触觉）；仿真九物体 z 轴基准 + x/y 轴扩展
- **开源状态：** **宣称将开源 / 待发布**（截至 **2026-09-11**：项目页写明 *We will release the codes soon!*；用户提供的 <https://github.com/sharpa-robotics/WM-Craftnet> 返回 **404**，无公开可运行仓库）
- **入库日期：** 2026-09-11
- **一句话说明：** Dreamer 式 **World Synesthesia Model（WSM）** 从本体、腕部 depth、触觉、动作与奖励学习 action-conditioned RSSM，用 **干净 depth 重建监督** 与多模态解码训练；部署时 **不做想象 rollout**，只把确定性循环特征 \(h_t\) 作为 asymmetric actor–critic 的 **可复用任务上下文**，在 pose 扰动、外力、未见物体与多旋转轴上实现鲁棒手内旋转。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://wmcraftnet.github.io/> | 消融表、真机/仿真视频、局限与 BibTeX |
| 项目页归档 | [wmcraftnet-github-io.md](../sites/wmcraftnet-github-io.md) | 开源核查与指标摘录 |
| 代码（待发布） | <https://github.com/sharpa-robotics/WM-Craftnet> | 截至入库日 **404**；见 [wm-craftnet.md](../repos/wm-craftnet.md) |
| 手内旋转基线 IHR | 项目页对照 | tactile + noisy depth；WSM 去噪 depth 可提升但仍远低于 WM-Craftnet |
| 世界模型行走对照 | [WM-LOCO](../../wiki/entities/paper-wm-loco.md) | 同类 RSSM 特征喂策略，但任务为 G1 落脚约束行走 |

## 摘要级要点

- **问题：** 视触觉 RL 手内旋转在受控设定下很强，但对 **位姿偏移、力扰动、物体变化** 鲁棒性不足；开环 finger gait 在分布外易卡死或掉落。
- **WSM：** Dreamer-style RSSM 重建 depth 与低维 proprio-tactile，预测 reward；**输入 noisy depth、监督 clean depth**，为真机提供去噪几何状态。
- **策略用法：** WSM **不作为** latent imagination 或 WM 内 policy optimization；确定性循环特征 \(h_t\) 作为 **recurrent task context** 给 PPO actor–critic。
- **仿真消融（z 轴，表 1 节选）：** 最佳 raw-sensor baseline Return **386.9**；from-scratch **414.3**；WSM prop-only **684.5**；prop+tac **698.9**；**prop+tac+depth 753.3**（RotR **1.293**，Fall **0.005**）。
- **真机 duck z 轴：** IHR **1.83 rad / 5/10**；IHR+WSM-denoised depth **2.76 / 8/10**；**WM-Craftnet 16.18 / 10/10**。
- **可迁移先验：** 九物体 z 轴预训练 WSM → 四十九物体下游策略：3000 epoch 后 **9.37±0.13 rad/ep** vs 无先验 **3.28**；fall rate **0.3%** vs **6%**。
- **多轴：** 除 z 外训练/评测 x、y 轴旋转；y 轴强调力臂与侧向力矩；x 轴依赖滚动与指尖重分配。
- **零样本：** 九物体训练策略在三种未见几何上仍部分成功（项目页视频）。
- **局限：** 定量仍以短视界手内旋转为主；平移、工具使用等多为定性；难初始位姿与长程接触漂移仍会失败。

## 核心摘录（面向 wiki 编译）

### 1) WSM 训练目标

多模态重建（clean depth、proprio、tactile）+ reward 预测；action-conditioned 潜变量动力学。Clean-depth head 优于 noisy-depth 监督（Key controls **+45.3** Return）。

### 2) 部署接口

推理阶段 actor 读取 WSM 的 **deterministic recurrent feature**（非开环想象轨迹）；与腕部 noisy depth、触觉、本体并联进入策略。

### 3) 单策略多物体

无 object ID；t-SNE 显示 \(h_t\) 按几何/接触工况聚类。一条闭环策略覆盖不同尺寸、质量、曲率物体。

### 4) 扰动恢复

过程中外力、OOD 初始位姿、掌区滑移等：策略用 visuotactile + 循环状态重居中/再抓握后再转。

### 5) 工具使用（定性）

螺丝刀目标位姿平移与快速轴向旋转，展示 elongated geometry 下的 in-hand 控制。

## 对 wiki 的映射

- [WM-Craftnet 论文实体](../../wiki/entities/paper-wm-craftnet.md)
- [In-hand Reorientation](../../wiki/methods/in-hand-reorientation.md)
- [Manipulation](../../wiki/tasks/manipulation.md)
- [Tactile Sensing](../../wiki/concepts/tactile-sensing.md)
- [Model-Based RL](../../wiki/methods/model-based-rl.md)
- [WM-LOCO](../../wiki/entities/paper-wm-loco.md) — RSSM 特征作策略上下文的对照（行走域）

## 参考来源（原始）

- arXiv：<https://arxiv.org/abs/2609.07002>
- 项目页：<https://wmcraftnet.github.io/>
