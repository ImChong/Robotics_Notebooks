# Bench2Dex（arXiv:2609.15726）

> 来源归档（paper）

- **标题：** Bench2Dex: Benchmarking Visuo-Tactile Bimanual Dexterous Manipulation Across Dexterous Hands
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.15726>
- **PDF：** <https://arxiv.org/pdf/2609.15726>
- **项目页：** <https://bench2dex.github.io/>
- **代码：** <https://github.com/Bench2Dex/Bench2Dex>
- **文档：** <https://bench2dex.github.io/doc/>
- **入库日期：** 2026-09-20
- **再核日期：** 2026-09-20（项目页 + GitHub 步骤 2.5）
- **一句话说明：** Isaac Lab 统一 visuotactile 双臂基准：12 种灵巧手、26 任务、~1.3K 遥操作 demo；7 类扰动分 invariance/equivariance 四通道；评测 ACT/DP/π0.5/GR00T N1.5。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-09-20）：训练、推理、遥操作代码 [Bench2Dex/Bench2Dex](https://github.com/Bench2Dex/Bench2Dex)；资产/数据/权重见 HF 与 ModelScope（[Assets](https://huggingface.co/datasets/Bench2Dex/Assets)、[teleopdata](https://huggingface.co/datasets/Bench2Dex/teleopdata)、[policy_ckpt](https://huggingface.co/Bench2Dex/policy_ckpt)）。

## 核心摘录

1. **问题：** 灵巧手触觉硬件未收敛，跨 hand morphology 缺一致 visuotactile 实验设定。
2. **接口：** 共享仿真触觉接口——局部接触几何 → 8-bit 类图像观测；**不**复现特定物理传感器。
3. **规模：** 12 hands × 26 bimanual 长程任务；~1.3K human teleop demos；8 模态同步 HDF5。
4. **泛化轴：** 7 扰动 → invariance（视觉无关）vs equivariance（几何相关）；None/Equi./Inv./Full 四通道。
5. **评测：** Stable SR + LSCR + 效率/安全；ACT/DP/π₀.₅/GR00T N1.5；None 条件下 GR00T N1.5 aggregate 最高，Full 组合偏移下各策略均退化。
6. **采集：** Manus + ARKit → DexPilot + Pinocchio IK；replay 离线生成 RGB/触觉。

**对 wiki 的映射**

- [paper-bench2dex](../../wiki/entities/paper-bench2dex.md)
- [bench2dex-github-io](../sites/bench2dex-github-io.md)
- [bench2dex](../repos/bench2dex.md)
