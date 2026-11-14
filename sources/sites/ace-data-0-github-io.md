# ACE-Data-0 项目页（ace-data-engine.github.io/ACE-Data-0）

> 来源归档

- **标题：** ACE-Data-0: Human-Centric Ambient Capture as Embodied Data Engine
- **类型：** site / project-page
- **URL：** <https://ace-data-engine.github.io/ACE-Data-0/>
- **论文：** <https://arxiv.org/abs/2607.28625> — 归档见 [`sources/papers/ace_data_0_arxiv_2607_28625.md`](../papers/ace_data_0_arxiv_2607_28625.md)
- **数据集：** <https://huggingface.co/datasets/ACERobotics/ACE-Data-0> — 归档见 [`sources/datasets/ace-data-0.md`](../datasets/ace-data-0.md)
- **机构：** S-Lab / 南洋理工大学（NTU）；大晓机器人（Ace Robotics / ACERobotics）
- **入库日期：** 2026-08-10（2026-11-14 核对页面完整 HTML）
- **核对范围：** 2026-11-14 通读 `https://ace-data-engine.github.io/ACE-Data-0/` 页面 HTML 源码全部 1,052 行，核对数据集按钮外链、硬件配置、5 类标注示例轮播、三类任务分支与三层 benchmark 叙事。
- **通读：** 已完成
- **一句话说明：** Ambient Capture Engine（ACE）与 ACE-Data-0 官方项目站：双尺度采集、五模态同步、分层 benchmark 叙事与 Demo；导航链到 arXiv 与 Hugging Face 数据集。

## 开源核查（步骤 2.5，截至 2026-11-14）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到代码 | **否** — 可见入口为 Hugging Face · dataset；未见 GitHub 训练/评测仓 |
| 项目页是否链到数据 | **是** → `ACERobotics/ACE-Data-0`（HF API 实测已上传 ~22.18 TB、`gated: false`，README 仍标 `Data_Files-Coming_Soon`） |
| 仓内可运行训练 / benchmark 入口 | **未见** |
| 综合判定 | **数据已上传 HF（当前 API `gated: false`，非商业研究许可）+ 训练/评测代码未列** → **部分开源** |

## 页面要点

- Hero：ACE 将真实家居变为校准同步的录制工作室；table-scale / room-scale 双配置。
- 模态栈：Ego（4 鱼眼 + IMU）/ Exo（多视角 RGB）/ 3D 运动（身体·手·物体 60 Hz）/ 触觉手套 / 多通道音频。
- 数据集叙事：150 h · 200 任务类 · 75k episodes · Atomic HOI / Chain of HOI / HSI。
- Benchmark：signals → scene components → interactions 三层诊断。

## 关联资料

- 论文摘录：[`sources/papers/ace_data_0_arxiv_2607_28625.md`](../papers/ace_data_0_arxiv_2607_28625.md)
- 数据集归档：[`sources/datasets/ace-data-0.md`](../datasets/ace-data-0.md)
- Wiki 实体：[`wiki/entities/paper-ace-data-0.md`](../../wiki/entities/paper-ace-data-0.md)
