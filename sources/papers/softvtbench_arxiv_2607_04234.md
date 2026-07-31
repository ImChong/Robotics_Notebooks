# SoftVTBench（arXiv:2607.04234）

> 来源归档（ingest）

- **标题：** SoftVTBench: A Safety-Aware Visuo-Tactile Benchmark for Physically Constrained Robotic Manipulation of Deformable Objects
- **类型：** paper / benchmark / visuo-tactile / deformable-object manipulation / safety evaluation
- **arXiv：** <https://arxiv.org/abs/2607.04234>（PDF：<https://arxiv.org/pdf/2607.04234.pdf>）
- **作者：** Bowen Jing\*、Mingxin Wang\*、Ruiyang Hao、Chenchen Ge、Hanwen Shen、Junjie He、Yang Cui、Yiming Hou、Weitao Zhou‡、Jiawei Wang、Minglei Li、Dandan Zhang、Ding Zhao、Houde Liu、Xiaofan Li、Si Liu、Ping Luo、Haibao Yu‡（\* equal；‡ corresponding）
- **机构：** 拓境智能（Tuojing Intelligence）；清华大学（Tsinghua）；伦敦国王学院（King's College London）；东南大学（Southeast University）；史蒂文斯理工学院（Stevens）；香港科技大学广州校区（HKUST-GZ）；曼彻斯特大学（University of Manchester）；简易人工智能（Simple AI）；帝国理工学院（Imperial College London）；卡内基梅隆大学（CMU）；浙江大学（ZJU）；北京航空航天大学（BUAA）；香港大学（HKU）
- **项目页：** <https://softvtbench.github.io/>
- **代码：** <https://github.com/TuojingAI/SoftVTBench>
- **数据集：** [Hugging Face `Arthur12137/SoftVTBench`](https://huggingface.co/datasets/Arthur12137/SoftVTBench) · [ModelScope `Arthur12137/SoftVTBench`](https://www.modelscope.cn/datasets/Arthur12137/SoftVTBench)
- **入库日期：** 2026-07-29
- **一句话说明：** Isaac Sim + FEM 可变形体上的 **安全感知视触觉操作基准**：分开报告 **Goal Success** 与 **Safety Success**，暴露「目标达成但过压/掉落」的虚假成功；π₀.₅ VO/VT 基线显示触觉主要抬高安全成功率而非刚体目标成功率。

## 开源状态（核查，2026-07-29）

- **已开源（代码 + 数据 + 资产包）：** 官方仓 [TuojingAI/SoftVTBench](https://github.com/TuojingAI/SoftVTBench)（Apache-2.0）含 Isaac Lab 扩展 `SoftVTBench/source/tac_manip`、OpenPI 训练/评测脚本、`tools/doctor.py`、安全阈值 `configs/safety_thresholds.json`；演示与 USD 资产在 HF/ModelScope。
- **项目页边界：** 页头 **Paper / Dataset** 按钮仍标 *coming soon*，但 **Code** 链到 GitHub；以 **README 徽章与 HF/ModelScope 镜像** 为准记数据集已发布（约 1,628 episodes 当前托管；论文口径 2,000）。
- **待发布：** SoftVTBench **参考 checkpoint** 计划外发，不入库 GitHub；需自训或等待权重。
- **上游依赖：** 仿真改编自 [Tabero](https://github.com/NathanWu7/Tabero)；Franka/GelSight 运行时资产见 [Tactile_Manipulation_Dataset](https://huggingface.co/datasets/china-sae-robotics/Tactile_Manipulation_Dataset)；π₀.₅ base 见 OpenPI。
- **交叉归档：** 项目页 [`sources/sites/softvtbench-github-io.md`](../sites/softvtbench-github-io.md)；代码 [`sources/repos/softvtbench.md`](../repos/softvtbench.md)。

## 摘要级要点

- **问题：** 可变形物体操作除「放到目标」外，还须 **稳抓不掉且不过度形变**；既有操作基准多为 success-only，掩盖不安全完成。
- **设定：** Isaac Sim + PhysX FEM soft body；Franka + 双指 GelSight Mini（RGB + marker motion）；第三人称/腕部 RGB + 本体 + 语言；20 Hz；四套件 2×2（Object/Spatial × Soft/Rigid）。
- **指标：** Goal Success vs Safety Success（后者 = Goal ∧ NoDrop ∧ \(D_{\mathrm{peak}}\le\tau_o\)，形变来自 **策略不可见** 的 FEM 特权态）。
- **基线：** π₀.₅-Vision（二进制夹爪）vs π₀.₅-Visuo-Tactile（触觉 RGB 历史 + marker + 连续夹爪）；LoRA 微调。
- **发现：** 刚体套件上触觉增益不一致；软体套件 Goal 接近，但 Safety 显著提升（Object-Soft 21.4%→35.6%；Spatial-Soft 32.6%→44.6%），形变分布整体下移。

## 核心论文摘录（MVP）

### 1) Goal–Safety 分离评测

- **链接：** §3.1、§3.4；Eq. (3)–(5)
- **摘录要点：** Safety Success 严于 Goal；峰值 FEM-RMS 形变相对包围盒对角线归一；\(\tau_o\) 由离线抓取–压缩标定。Gap 量化「虚假成功」。
- **对 wiki 的映射：**
  - [SoftVTBench 实体页](../../wiki/entities/paper-softvtbench.md)
  - [具身评测基准选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)
  - [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)

### 2) 匹配 2×2 任务套件 + 视触觉观测

- **链接：** §3.2–3.3；Table 2；Appendix B
- **摘录要点：** Object/Spatial Soft 评可变形安全；匹配 Rigid 作诊断对照。观测含双视角 RGB、双指触觉 RGB+marker、本体、语言；动作绝对 EE 位姿 + 夹爪。
- **对 wiki 的映射：**
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)
  - [Tactile Sensing](../../wiki/concepts/tactile-sensing.md)
  - [Manipulation](../../wiki/tasks/manipulation.md)

### 3) 触觉主要改善软体安全而非刚体 Goal

- **链接：** §4；Table 3–4
- **摘录要点：** Object-Soft Safety 21.4%→35.6%；Spatial-Soft 32.6%→44.6%；刚体 Object-Rigid VT 反低于 VO。形变 mean/median/P95 全面下降。
- **对 wiki 的映射：**
  - [SoftVTBench 实体页](../../wiki/entities/paper-softvtbench.md)
  - [触觉专题](../../wiki/overview/hub-tactile.md)
  - [TacO 传感器基准](../../wiki/entities/paper-taco-tactile-sensor-benchmark.md)（互补：硬件选型 vs 安全指标）

## BibTeX

```bibtex
@article{jing2026softvtbench,
  title   = {SoftVTBench: A Safety-Aware Visuo-Tactile Benchmark for
             Physically Constrained Robotic Manipulation of Deformable Objects},
  author  = {Jing, Bowen and Wang, Mingxin and Hao, Ruiyang and Ge, Chenchen and
             Shen, Hanwen and He, Junjie and Cui, Yang and Hou, Yiming and
             Zhou, Weitao and Wang, Jiawei and Li, Minglei and Zhang, Dandan and
             Zhao, Ding and Liu, Houde and Li, Xiaofan and Liu, Si and
             Luo, Ping and Yu, Haibao},
  journal = {arXiv preprint arXiv:2607.04234},
  year    = {2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-softvtbench.md`](../../wiki/entities/paper-softvtbench.md)
- 项目页归档：[`sources/sites/softvtbench-github-io.md`](../sites/softvtbench-github-io.md)
- 代码归档：[`sources/repos/softvtbench.md`](../repos/softvtbench.md)
- 互链：[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)、[Tactile Sensing](../../wiki/concepts/tactile-sensing.md)、[接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)、[触觉专题](../../wiki/overview/hub-tactile.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[TacO](../../wiki/entities/paper-taco-tactile-sensor-benchmark.md)、[具身评测基准选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)、[VLA](../../wiki/methods/vla.md)
