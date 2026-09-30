# WB-WAM 项目页（wb-wam.github.io）

- **类型：** 项目静态站点
- **收录日期：** 2026-09-30
- **站点：** <https://wb-wam.github.io/>
- **论文：** <https://arxiv.org/abs/2609.34199>
- **机构（页脚）：** Tsinghua University；Xiong'an Institute of Artificial Intelligence；The University of Melbourne

## 一句话

**WB-WAM** 把 **body / root / dexterous hand** 监督写进 **生成式 video 预训练**，经 **异构 1880.2 h → PICO 22 h → 真机 3.37 h** 三阶段落地 **G1 + Wuji + SONIC** 人形 loco-manipulation。

## 开源核查（2026-09-30）

| 项 | 结论 |
|----|------|
| **代码** | **待发布** — 页内 `resource-github` 按钮 **`disabled`**，无 GitHub URL |
| **权重 / 数据** | **待发布** — `resource-huggingface` 按钮 **`disabled`** |
| **论文** | **已公开** — arXiv:2609.34199 |

## 站点摘录要点

- **规模：** 预训练 **1,880.2 h**；**72-D** physical action space；**3** 训练阶段。
- **执行栈：** body/root → **SONIC**；hand → **Wuji** 指关节直接控制。
- **WB-Datasets：** Stage II **22 h / 73 tasks / 13,396 ep** PICO；Stage III **3.37 h / 8+2 tasks / 1,011 ep** SONIC 遥操作。
- **仿真：** **HumanoidArena** 七任务，报告 **81.9%** 均值 SR（SONIC 设定）。
- **真机：** **8** 任务视频展示；定量 **五任务 84.0%** 均值 SR；**2** 任务 **5 连成功** 片段。

## 对 wiki 的映射

- 主沉淀：[WB-WAM](../../wiki/entities/paper-wb-wam.md)
- 论文归档：[wb_wam_arxiv_2609_34199.md](../papers/wb_wam_arxiv_2609_34199.md)
