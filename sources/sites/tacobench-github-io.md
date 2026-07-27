# tacobench.github.io（TacO 项目页）

- **标题：** TacO: Benchmarking Tactile Sensors for Object Manipulation
- **类型：** site / project-page
- **URL：** <https://tacobench.github.io/>
- **入库日期：** 2026-07-27
- **配套论文：** [TacO（arXiv:2605.21976）](https://arxiv.org/abs/2605.21976) — 归档见 [`sources/papers/taco_tactile_sensor_benchmark_arxiv_2605_21976.md`](../papers/taco_tactile_sensor_benchmark_arxiv_2605_21976.md)
- **代码：** <https://github.com/TacObench/TacO>
- **硬件：** <https://github.com/TacObench/TacObench.github.io/tree/main/3D_part_files>
- **名称消歧：** 本页 **TacO（传感器基准）** ≠ [taco-wm.github.io](https://taco-wm.github.io/)（触觉 WM 自我纠正 / VLA 后训练）。

## 一句话摘要

官方项目页展示 **六传感器 × 四模态** 对比、基于 ACT 的标准化模仿学习管线，以及 pick-and-place / 插入 / 重定向三项真机实验视频与跨传感器材料摩擦分析。

## 公开信息要点（截至入库日）

- **核心叙事：** 没有一种通用最佳触觉传感器（含昂贵视觉触觉）；正确选择取决于任务；希望帮助社区评估下一代传感器及其对视觉策略的增益。
- **传感器板块：** Visual / Acoustic / Magnetic / Resistive；覆盖空间分辨率、剪切力、表征形式等能力差异。
- **策略板块：** 每传感器训 **visuotactile** 与 **vision-only** 两套 ACT；模态特异编码器（MLP / 卷积 / 频谱）。
- **实验视频：** 未知质量抓取放置、遮挡插头插入、连续调力重定向；强调滑移/振动检测相对纯法向力的优势，以及连续调力任务上低成本传感器可与高成本相当。
- **可重复性：** 提供开源测试夹具 3D 文件与重复性测试说明。
- **开源边界：** 页上可进 **Code** 与 **3D_part_files**；**示范数据 / 权重下载链接截至入库日未在项目页头部显式列出**（与论文「will be publicly available」表述对照，按部分开源记录）。

## 为何值得保留

- **非 PDF 证据：** 真机失败/成功视频比表格更直观呈现「触觉何时真的救命」。
- **选型入口：** 传感器硬件与策略管线同页，适合工程选型时快速扫一眼模态差异。

## 关联资料

- 论文归档：[`sources/papers/taco_tactile_sensor_benchmark_arxiv_2605_21976.md`](../papers/taco_tactile_sensor_benchmark_arxiv_2605_21976.md)
- 代码归档：[`sources/repos/taco-bench.md`](../repos/taco-bench.md)
- wiki：[`wiki/entities/paper-taco-tactile-sensor-benchmark.md`](../../wiki/entities/paper-taco-tactile-sensor-benchmark.md)
