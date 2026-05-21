# EgoScale：Scaling Dexterous Manipulation with Diverse Egocentric Human Data

> 来源归档（ingest）

- **标题：** EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data
- **类型：** paper
- **机构：** NVIDIA GEAR；UC Berkeley；University of Maryland
- **原始链接：**
  - <https://arxiv.org/abs/2602.16710>
  - <https://arxiv.org/html/2602.16710v1>（HTML 版便于锚点跳转）
  - 项目页：<https://research.nvidia.com/labs/gear/egoscale/>
- **入库日期：** 2026-05-17
- **一句话说明：** 在 **2 万小时以上**、带 **腕部 + 重定向高 DoF 手部动作标签** 的第一人称人视频上预训练 **流式 VLA**，发现 **人数据规模与验证损失呈 log-linear 缩放律**，且该离线指标与 **真机灵巧长程操作** 强相关；再用 **小规模人–机视点/场景对齐的 mid-training** 把表示锚到机器人感知与控制，配合任务后训练实现 **相对无预训练基线约 +54% 平均成功率** 与 **极少机器人演示下的一跳/少跳泛化**。

## 核心论文摘录（MVP）

### 1) 问题：灵巧人–机迁移是否随「人数据规模」可预测地变好

- **链接：** <https://arxiv.org/html/2602.16710v1#S1>
- **摘录要点：** 既有工作多限于 **数十到数百小时** 人数据或 **低 DoF 末端**；尚不清楚 **大规模 egocentric 人视频** 能否支撑 **高 DoF 手指级** 灵巧操作。论文主张：**有效的人–灵巧机迁移在测量范围内表现为 scaling 现象**。
- **对 wiki 的映射：**
  - [EgoScale](../../wiki/methods/egoscale.md) — 将「人侧数据小时」与 **可重复的离线验证指标 → 真机表现** 的实证链条写清楚。

### 2) 人侧动作表示：相对腕部 SE(3) + 优化式重定向到 22-DoF 手关节

- **链接：** <https://arxiv.org/html/2602.16710v1#S2.SS1>
- **摘录要点：** 用 SLAM / 手部关键点估计得到相机与 21 点手部 SE(3)；**腕在世界的相对增量** \(\Delta\mathbf{W}\) 作为 **与人、机共享** 的臂级抽象，减弱全局相机漂移；手指侧把关键点 **重定向** 到目标灵巧手（默认 **Sharpa 22-DoF**）关节空间，使预训练监督直接落在 **可迁移的操纵运动结构** 上。
- **对 wiki 的映射：**
  - [EgoScale](../../wiki/methods/egoscale.md) — 「显式运动监督 vs 纯视觉表征」与 [Motion Retargeting](../../wiki/concepts/motion-retargeting.md) 在数据管线中的接口位置。

### 3) 数据：Stage I 大规模野外 egocentric + EgoDex 精信号；Stage II 对齐人–机 play

- **链接：** <https://arxiv.org/html/2602.16710v1#S2.SS2>
- **摘录要点：** Stage I 合计约 **20,854 h**（论文叙述的多数为野外 egocentric，**829 h** 为 **EgoDex** 高精度补充）；Stage II **344** 桌面任务、每任务约 **30** 条人轨与 **5** 条机轨，约 **50 h** 人 + **4 h** 机，**机位/内参与机器人一致**（头显 + 双腕相机；人用 Vive + Manus 与遥控栈同步）。
- **对 wiki 的映射：**
  - [EgoScale](../../wiki/methods/egoscale.md) — **规模（diversity）与对齐（correspondence）解耦** 的两阶段数据设计读点。

### 4) 模型与三阶段训练：GR00T N1 式流 VLA；人数据用可学习本体占位 token

- **链接：** <https://arxiv.org/html/2602.16710v1#S2.SS3>–<https://arxiv.org/html/2602.16710v1#S2.SS4>
- **摘录要点：** **VLM 骨干 + DiT 动作专家 + flow matching**；人轨无本体时以 **learnable placeholder** 代替 \(q_t\)；跨本体用 **轻量 embodiment adapter** 处理 proprio 与手部输出，**腕流 + 共享骨干** 跨形态复用。
- **对 wiki 的映射：**
  - [EgoScale](../../wiki/methods/egoscale.md) — 与通用 [VLA](../../wiki/methods/vla.md) 叙述对齐：何处共享、何处 per-embodiment。

### 5) 缩放律与 mid-training 的实证结论（摘要级）

- **链接：** <https://arxiv.org/html/2602.16710v1#S3>
- **摘录要点：** 在 **1k–20k h** 人预训练扫描上，**最优人轨验证 MSE 与数据规模 log 线性拟合 \(R^2\approx 0.998\)**，并与 **后训练后真机平均完成度** 同向单调；**人预训练 + 对齐 mid-training** 在 **单条机演示 + 对齐人演示** 的 one-shot 设定下显著优于仅 mid 或仅人预训练；人预训练表示可迁移到 **Unitree G1 三指手** 等 **低 DoF** 形态（论文给出绝对成功率增益叙述）。
- **对 wiki 的映射：**
  - [EgoScale](../../wiki/methods/egoscale.md) — 与 [具身规模法则](../../wiki/concepts/embodied-scaling-laws.md)、[HumanNet](../../wiki/entities/humannet.md) 并列讨论「**人视频小时**作为可扩展监督」时的 **指标选择与对齐成本**。

## 当前提炼状态

- [x] 摘要、§2 表示/数据/架构/训练日程、§3 主结果与缩放、one-shot / 跨本体段落已摘录到可维护粒度
- [x] 与 `sources/sites/nvidia-research-egoscale.md` 分工：公式与阶段细节以 arXiv HTML 为准；NVIDIA 项目页侧重对外表述、作者致谢与 BibTeX

## BibTeX（项目页一致）

```bibtex
@misc{zheng2026egoscalescalingdexterousmanipulation,
      title={EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data},
      author={Ruijie Zheng and Dantong Niu and Yuqi Xie and Jing Wang and Mengda Xu and Yunfan Jiang and Fernando Castañeda and Fengyuan Hu and You Liang Tan and Letian Fu and Trevor Darrell and Furong Huang and Yuke Zhu and Danfei Xu and Linxi Fan},
      year={2026},
      eprint={2602.16710},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.16710},
}
```
