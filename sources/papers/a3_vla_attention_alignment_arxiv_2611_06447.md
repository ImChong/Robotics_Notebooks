# A³-VLA：Automatic Perception-Guided Attention Alignment for Robot Manipulation（arXiv:2611.06447）

> 来源归档（paper）

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.06447>（核对时仅有 v1；`/abs/2611.06447v2` 返回 404）
- **PDF：** <https://arxiv.org/pdf/2611.06447>（v1，cs.RO，2026-11-04 提交；PDF 页眉标注 NeurIPS 2026 第 40 届会议格式）
- **标题（arXiv 元数据）：** $A^3$-VLA: Automatic Perception-Guided Attention Alignment for Robot Manipulation
- **作者：** Shengzhe Zhang（中国科学技术大学）、Qi Zhang（上海人工智能实验室）、Dazhong Shen（南京航空航天大学）、Chao Wang（中国科学技术大学；认知智能全国重点实验室）、Shaopeng Zhai（上海人工智能实验室）、Hui Xiong（香港科技大学（广州）/香港科技大学）；通讯作者为 Qi Zhang、Chao Wang、Hui Xiong
- **代码 / 项目页：** 论文与 PDF 元数据中均**未给出 URL**；NeurIPS Checklist 第 13 条称代码放在「anonymous GitHub repository」，但正文无链接。本次核查未找到公开仓库（见 [sources/sites/a3-vla-attention-alignment.md](../sites/a3-vla-attention-alignment.md)）
- **入库日期：** 2026-11-06
- **线索来源：** [2026-11-06-seven-robotics-intake-draft.md](2026-11-06-seven-robotics-intake-draft.md)

## 核心摘录（人工转述，已通读正文 §1–§5 与附录 A.1–A.8、Checklist）

- 问题：VLA 的动作损失只监督动作精度，不监督视觉注意力是否落在任务相关区域；微调中 VLM 骨干的注意力会漂移到与任务无关的视觉捷径（Fig.1：π0、OpenVLA-OFT 在 VLABench Select Poker 上指令→图像注意力落在错误目标）。
- 方法：两阶段。(a) 离线：用交互时刻 t_k 前后各 w 帧的逐 patch 帧间差，筛出「交互前稳定、交互后变化」的 patch 作为目标物候选 C_tar，用「附近帧高变化」patch 作为机器人候选 C_rob（τ=10），再以 SAM2 把 patch 掩码作为 mask prompt 沿轨迹传播（A.6），映射回 VLA patch 网格（重叠比 ρ=0.25）。(b) 微调：L = L_FM + λ1(L_l2i + L_s2i) + λ2 L_cl（式 10），λ1=0.1、λ2=0.01。
- L_l2i / L_s2i：对最后一层、头平均后的注意力，要求语言 token（取平均）/ state token 分到候选区域的注意力质量不低于阈值 η（均 0.25），平方 hinge 形式（式 6、7）。
- L_cl：语言 token 均值为锚，C_tar 内 patch 均值为正，其余 patch 均值为负，二元 InfoNCE（余弦相似度，式 8–9），仅用于交互前帧。
- 推理：只用微调后的 VLA，无额外开销；SAM2 掩码离线预计算缓存。
- 骨干：π0、OpenVLA-OFT、π0.5；训练 8×H200；π0 / π0.5 batch 32 / 256、30k 步；OpenVLA-OFT 每卡 8、50k 步（A.5）。
- 结果（表号见 wiki 页）：VLABench 平均 π0 0.47→0.64、OpenVLA-OFT 0.16→0.40、π0.5 0.41→0.67（Table 1）；RoboTwin 2.0 平均 0.42→0.52、0.20→0.27、0.58→0.65（Table 2）；真机 SR 平均 π0 0.55→0.76（Table 3）。
- 局限（A.7）：目标被完全遮挡、与背景难分、过小、与干扰物一同运动时伪区域变差；需要粗交互时刻（夹爪状态、接触信号或预设状态区间）；SAM2 离线预处理增加数据准备成本并依赖其质量。
- Checklist：未报告误差条（"No"，理由为每任务评测数十次取平均）。

## 对 wiki 的映射

- [paper-a3-vla-attention-alignment](../../wiki/entities/paper-a3-vla-attention-alignment.md)
