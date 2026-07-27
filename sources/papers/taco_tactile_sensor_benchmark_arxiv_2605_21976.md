# TacO: Benchmarking Tactile Sensors for Object Manipulation（arXiv:2605.21976）

> 来源归档（ingest）

- **标题：** TacO: Benchmarking Tactile Sensors for Object Manipulation
- **类型：** paper / tactile sensing / benchmark / imitation learning / real-world manipulation
- **arXiv：** <https://arxiv.org/abs/2605.21976>（PDF：<https://arxiv.org/pdf/2605.21976.pdf>）
- **作者：** Anya Zorin、Zilin Si、Myungsun Park、Junsung Park、Alexiy Buynitsky、Sachin Bhadang、Taejun Park、Sohee John Yoon、Yong-Lae Park、Oliver Kroemer、Zeynep Temel、Michael T. Tolley、Sha Yi、Xiaolong Wang
- **机构：** 加州大学圣地亚哥分校（UCSD）；卡内基梅隆大学（CMU）；首尔大学（SNU）
- **项目页：** <https://tacobench.github.io/>
- **代码：** <https://github.com/TacObench/TacO>
- **硬件（3D 零件）：** <https://github.com/TacObench/TacObench.github.io/tree/main/3D_part_files>
- **入库日期：** 2026-07-27
- **一句话说明：** 用统一 ACT 模仿学习管线，在 **6 种 / 四模态** 触觉传感器 × **3 项真机操作任务** 上做跨传感器比较；结论是 **没有通用最佳触觉传感器**，选型取决于任务与材料摩擦等 embodiment 因素。
- **名称消歧：** 本页 **TacO（触觉传感器基准）** ≠ 库内已有 [TACO · 触觉世界模型作自我纠正器](../../wiki/entities/paper-taco-tactile-wm-vla-posttrain.md)（arXiv:2607.02840，taco-wm.github.io）。

## 开源状态（核查，2026-07-27）

- **已开源（代码 + 硬件）：** 官方仓 [TacObench/TacO](https://github.com/TacObench/TacO) 含 `tactile_policy/`（ACT 训练 / 远程推理）、`tactile_sensors/`、`hardware_repeatability/`；项目页仓库 [3D_part_files](https://github.com/TacObench/TacObench.github.io/tree/main/3D_part_files) 提供夹爪 / 传感器安装 STL 与可重复性测试夹具。
- **可运行入口：** `bash create_env.sh` → `python tactile_policy/main.py --model_cfg … --dataset_json …`；部署见 `tactile_policy/remote_inference/serve_act_policy.py` / `export_act_to_jit.py`。
- **边界（部分）：** 论文与项目页宣称 **code / data / hardware** 公开；截至入库日 **仓库 README 与项目页未给出示范数据集或 checkpoint 的公开下载 URL**（仅规定 HDF5 格式与目录约定）。复现需自备遥操作数据或等待数据发布。
- **交叉归档：** 项目页 [`sources/sites/tacobench-github-io.md`](../sites/tacobench-github-io.md)；代码 [`sources/repos/taco-bench.md`](../repos/taco-bench.md)。

## 摘要级要点

- **问题：** 社区公认触觉有助于接触丰富操作，但缺 **任务驱动、跨模态、真机** 的传感器选型证据；既有基准多为单模态或偏感知/重建。
- **设定：** 六传感器 — **FSR / FlexiTac / eGain**（电阻）、**eFlesh**（磁）、**Daimon**（视觉触觉）、**Contact Mic**（声学）；三任务 — **未知质量 pick-and-place**、**遮挡下插头插入**、**需连续调力的物体重定向**。
- **对照：** 同一批数据上训 **vision-only** 与 **visuotactile** 两套 ACT，隔离触觉信号贡献；另用 vision-only 跨传感器比较隔离 **材料/外形** 的 embodiment 效应。
- **发现：** 触觉通常相对 vision-only 提升成功率，但 **增益因任务与模态而异**；插入任务中带 shear / 振动的传感器增益更大；连续调力任务上 **低成本传感器可与高成本相当**；高摩擦材料整体更有利，但重定向中低摩擦在 vision-only 下反而更好。

## 核心论文摘录（MVP）

### 1) 任务驱动的跨模态触觉基准

- **链接：** §1；Table 1；§5
- **摘录要点：** 相对 ObjectFolder / Tactile MNIST / VTDexManip / ManiFeel，TacO 覆盖 **Acoustic + Magnetic + Resistive + Visual** 四模态、**6** 传感器，并配套开源可重复性测试套件与三项真机操作任务。
- **对 wiki 的映射：**
  - [TacO 触觉传感器基准](../../wiki/entities/paper-taco-tactile-sensor-benchmark.md)
  - [Tactile Sensing](../../wiki/concepts/tactile-sensing.md)

### 2) 模态特异编码器 + ACT 统一策略头

- **链接：** §4；§4.1
- **摘录要点：** 腕部 + 第三人称 RGB（ResNet18）与本体状态线性投影；触觉按模态用 MLP / ResNet / mel-spectrogram；CVAE ACT，chunk \(H=64\)，部署执行前 32 步。
- **对 wiki 的映射：**
  - [Action Chunking](../../wiki/methods/action-chunking.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)

### 3) 「无通用最佳传感器」与材料摩擦效应

- **链接：** §6；Table 3–5；§7
- **摘录要点：** 插入任务 Contact Mic / eFlesh **0.2/0.3 → 0.7**；重定向上廉价 FSR 与昂贵 Daimon 触觉策略成功率接近；高空间分辨率未必翻译为操作成功率；高摩擦表面在 vision-only 下多数任务更优。
- **对 wiki 的映射：**
  - [TacO 触觉传感器基准](../../wiki/entities/paper-taco-tactile-sensor-benchmark.md)
  - [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)
  - [触觉专题](../../wiki/overview/topic-tactile.md)

## BibTeX

```bibtex
@preprint{TacO2026,
  title={TacO: Benchmarking Tactile Sensors for Object Manipulation},
  author={Zorin, Anya and Si, Zilin and Park, Myungsun and Park, Junsung and
          Buynitsky, Alexiy and Bhadang, Sachin and Park, Taejun and Yoon, Sohee John and
          Park, Yong-Lae and Kroemer, Oliver and Temel, Zeynep and Tolley, Michael T. and
          Yi, Sha and Wang, Xiaolong},
  year={2026},
  url={https://arxiv.org/abs/2605.21976}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-taco-tactile-sensor-benchmark.md`](../../wiki/entities/paper-taco-tactile-sensor-benchmark.md)
- 项目页归档：[`sources/sites/tacobench-github-io.md`](../sites/tacobench-github-io.md)
- 代码归档：[`sources/repos/taco-bench.md`](../repos/taco-bench.md)
- 互链：[Tactile Sensing](../../wiki/concepts/tactile-sensing.md)、[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)、[Action Chunking](../../wiki/methods/action-chunking.md)、[Imitation Learning](../../wiki/methods/imitation-learning.md)、[触觉专题](../../wiki/overview/topic-tactile.md)、[VTAP Gripper](../../wiki/entities/paper-vtap-gripper.md)（同用 FlexiTac）、[TACO WM（消歧）](../../wiki/entities/paper-taco-tactile-wm-vla-posttrain.md)
