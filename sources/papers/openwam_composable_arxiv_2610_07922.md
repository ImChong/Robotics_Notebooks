# OpenWAM（Stanford，可组合 WAM 框架）论文来源归档

## 元数据

- **题目：** OpenWAM: An Open Framework for Composable World-Action Models
- **作者：** Heng Yu、David D. Yuan、Juze Zhang（共同一作）、Changan Chen、Yao Feng、Michelle Baldonado、Steve Cousins、Li Fei-Fei、Jiajun Wu、Ehsan Adeli
- **单位：** Stanford University（作者栏仅列该单位）
- **预印本：** https://arxiv.org/abs/2610.07922
- **HTML：** https://arxiv.org/html/2610.07922v1
- **版本：** v1，2026-10-06，cs.RO（交叉 cs.CV）
- **项目页：** https://openwam.stanford.edu （归档见 [sources/sites/openwam-stanford.md](../sites/openwam-stanford.md)）
- **代码：** https://github.com/OpenWAM/OpenWAM （归档见 [sources/repos/openwam_stanford.md](../repos/openwam_stanford.md)）
- **注意：** 与 NUS/清华等的 [OpenWAM（arXiv:2609.07398）](openwam_arxiv_2609_07398.md) 是**同名不同论文**，仓库也不同（`OpenWAM/OpenWAM` vs `OpenWAM-Official/OpenWAM`）。

## 论文要解决的问题

现有世界–动作模型（WAM）往往同时改动视频生成骨干、动作网络、模态交互方式与推理流程，难以判断收益究竟来自哪一项；世界模型与动作模型绑得过紧，也让组件难以跨任务复用。OpenWAM 要提供一个共享的因果机器人视频底座，让「交互方式」与「局部动力学（正/逆）」成为可替换的程序，在同一骨干上比较。

## 方法与训练信息

1. **底座：** 从 Wan2.2-5B 出发，在 >10,000 小时无动作标注的机器人与人–物交互视频上做因果（chunk-causal）继续预训练；语料包括 UMI/MV-UMI、AgiBot World、RoboMIND、InternData-A1、RoboCOIN、FastUMI、Ego-Exo4D、Open X-Embodiment。
2. **因果机制：** 未来视频 chunk 可看当前与已有 chunk，不看后续 chunk；chunk 内 token 联合去噪。视频以 4 帧潜变量为一个 chunk，每个 chunk 对应 16 步动作块；动作为末端位置 + 轴角 + 夹爪，本体感觉作为每 chunk 条件。
3. **动作分支：** 独立的动作 Transformer，经 Mixture-of-Transformers 与视频分支连接：各模态保留自己的归一化/投影/FFN，共享一次注意力；文本经独立 cross-attention。
4. **四种交互程序：** Joint（视频与动作同时去噪、双向注意力）、VTA（先视频后动作）、ATV（先动作后视频）、Decoupled（无未来跨模态注意力）。顺序模式训练时用真实轨迹做 teacher forcing，推理时用第一遍生成的轨迹。
5. **损失：** 潜空间 flow matching（噪声与目标潜变量线性插值，τ~U[0,1]，MSE 回归速度）；双目标程序损失相加。
6. **局部动力学：** IDM 建模 p(A⁺ | O⁺, O⁰, q_t)，FDM 建模 p(O⁺ | A⁺, O⁰, q_t)，均不使用语言、历史动作与更早观测，因而可跨任务复用。
7. **反事实数据（LIBERO-Long-CF）：** 从 LIBERO-Long 演示恢复仿真器状态，执行替代动作续接；同一初始状态的分支落在同一 train/val/test 切分；混合训练 η=0.6（60% 反事实 + 40% 演示）。项目页称 32,000 个反事实片段、覆盖十个任务。
8. 训练算力（GPU 数、步数、批大小）位于附录 B.3，本次抓取未读到，原文未核实。

## 论文报告的结果

- **LIBERO 四套件（3 seeds 均值）：** OpenWAM-VTA 98.6（Object 99.4 / Goal 98.4 / Spatial 98.6 / Long 97.8）；Decoupled 98.0；Joint 97.6；ATV 96.8。对照（已发表数字）：LingBot-VA 98.5、Motus 97.7、Fast-WAM 97.6、π0.5 96.9、OpenVLA-OFT 97.1、π0 94.1、GR00T-N1 93.9。
- **因果机器人视频适配消融（LIBERO-Long, VTA）：** 原始 Wan2.2-5B 初始化 68.4% → 机器人视频因果预训练骨干 97.8%（+29.4）；论文说明这是「机器人视频预训练 + 因果适配」的合并效应，二者未分离。
- **真机双臂（2×Franka Research 3，两腕部 + 一第三人称 RGB）：** Toast 92.0/90.0、Cube 90.0/94.0、Cups 94.4/91.7（VTA / Joint），均值 92.1 / 91.9；ATV、Decoupled 未报告真机。
- **冻结 IDM 迁移到 4 个 LIBERO-90 目标任务（任务微调视频预测器）：** 局部上下文 IDM 仅演示监督 21.5% → 纯反事实 84.5%、混合 84.0%；完整上下文 IDM 仅 44.5%/47.0%；未适配的原始 VTA 10.5%；任务微调 VTA 参考 93.0%。
- **LIBERO-Long 源任务组合：** 完整 IDM 演示 97.8；局部 IDM 演示 25.8，反事实 90.6，混合 94.4。
- **FDM（2,560 个反事实未来，160 个上下文 × 16 分支）：** 仅演示 RGB MSE 14.35×10⁻³；纯反事实 9.40×10⁻³（降 34.5%）；混合 9.62×10⁻³；K=16 身份识别准确率 21.1% → 71.3%（纯反事实）/ 67.7%（混合）。

## 复现与解释边界

- 论文自述局限：可组合性主要在仿真中验证；目标任务实验会适配视频预测器，完整视频–动作模型的零样本迁移未证明；单一多目标检查点未证明在所有模式上追平最强专用变体；FDM 仅覆盖局部短时程。
- 代码与预训练视频权重已公开；策略/IDM 检查点与 LIBERO-Long-CF 数据集截至 2026-10-12 仓库称「准备发布」。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-openwam-composable-world-action-models.md
- 概念入口：wiki/concepts/world-action-models.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.07922>
- arXiv HTML：<https://arxiv.org/html/2610.07922v1>
- 项目页：<https://openwam.stanford.edu>
- 代码：<https://github.com/OpenWAM/OpenWAM>
