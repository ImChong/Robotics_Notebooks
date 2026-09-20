# XPACE 官方项目页（XPENG Robotics）

> 来源归档

- **标题：** XPACE — Joint World and Action Modeling from Heterogeneous Experience
- **类型：** site / project-page
- **URL：** <https://xpeng-robotics.github.io/xpace/>
- **关联论文：** <https://arxiv.org/abs/2609.17372>
- **PDF：** <https://xpeng-robotics.github.io/xpace/assets/papers/xpace.pdf>
- **代码：** 项目页导航链至组织 <https://github.com/xpeng-robotics>；**截至 2026-09-20 无 XPACE 专用仓库或权重链接**
- **机构：** XPENG Robotics（小鹏机器人）
- **入库日期：** 2026-09-20
- **一句话说明：** XPACE 统一 **WAM + world simulator**：共享 video backbone 联合预测动作与未来视频，或按骨架控制预测视觉；5000h 异构经验 + 粗到细课程 + SGF 仿真自改进；IRON 人形真机验证。

## 步骤 2.5 开源核查（2026-09-20）

| 项 | 结论 |
|----|------|
| 项目页 Code / GitHub | 页眉链至 **xpeng-robotics** 组织，无 `xpace` 仓库入口 |
| 组织公开仓库 | DIAL、UniT、AnyWorld、xpeng-robotics.github.io 等；**不含 XPACE** |
| 权重 / 数据 | 页内未列 Hugging Face / ModelScope / Zenodo |
| **判定** | **未开源** — 技术报告与演示可公开访问，训练/推理/部署代码与 checkpoint **未发布** |

## 页面结构归纳

1. **Overview：** WAM 与 simulator 双模式、人–机 bridge、deviation–recovery + DAgger 自改进闭环（Figure 1）。
2. **Qualitative demos：** 语言指令切换目标（绿/红苹果）；人视频技能迁移（关抽屉、面包转移、叠碗）。
3. **Method：**
   - **Data：** L1 无动作 egocentric 视频、L2 人视频–动作、L3 bridge、L4 IRON 遥操作；F 失败/恢复集 **仅训 simulator**。
   - **Architecture：** 非对称 MoT；共享 causal Video Transformer；Action Transformer 读多级 video feature 预测 16-step action chunk；simulation 模式 skeleton + camera pose 条件、无 action 分支。
   - **Training：** Stage I 无动作视频适配；Stage II 视频–动作 / 仿真等概率 + flow matching，三阶段 human→bridge→robot 粗到细；Stage III 分 simulator（SGF）与 policy（8% 合成 recovery + 92% 原配方）两支。
4. **Results：**
   - 仿真：token addition skeleton 条件优于 AdaLN；SGF 长程 PSNR/latency 显著改善。
   - 真机（IRON-R01-1.11，20 trials/task）：平均成功率 XPACE **68.3%** vs DreamZero **40.0%** vs GR00T **6.7%**；叠碗任务 **不在机器人示范中** 但人/bridge 有覆盖。
   - 消融：Stage I 视频预训练 −12.5% action loss；human–robot co-training −14.0% vs robot-only；8% recovery DAgger 后成功率 **61.7%→86.7%**。

## 对 wiki 的映射

- 与 [`sources/papers/xpace_arxiv_2609_17372.md`](../papers/xpace_arxiv_2609_17372.md) 互为补充：论文/arXiv 偏摘要，本页偏 **数据金字塔、训练阶段、定量表与 demo 索引**。
- 沉淀：[`wiki/entities/paper-xpace.md`](../../wiki/entities/paper-xpace.md)
