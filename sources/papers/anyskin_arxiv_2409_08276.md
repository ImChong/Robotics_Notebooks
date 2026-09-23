# AnySkin: Plug-and-play Skin Sensing for Robotic Touch

> 来源归档（ingest）

- **标题：** AnySkin: Plug-and-play Skin Sensing for Robotic Touch
- **类型：** paper / hardware / tactile-sensing / magnetic-skin / replaceability / policy-learning
- **会议：** ICRA 2025（IEEE International Conference on Robotics and Automation）
- **arXiv abs：** <https://arxiv.org/abs/2409.08276>
- **arXiv HTML：** <https://arxiv.org/html/2409.08276v1>
- **PDF：** <https://arxiv.org/pdf/2409.08276>
- **DOI：** <https://doi.org/10.1109/ICRA55743.2025.11128638>
- **项目页：** <https://any-skin.github.io/>
- **机构：** New York University；Carnegie Mellon University；Columbia University；Meta AI Research
- **通讯作者：** Raunaq Bhirangi（raunaqbhirangi@nyu.edu）
- **提交：** 2024-09-12（arXiv v1）
- **入库日期：** 2026-09-23
- **arXiv 勘误：** 旧策展索引误写 **2401.17695**（与 AnySkin 无关）；正确编号为 **2409.08276**。
- **一句话说明：** 在 ReSkin 磁触觉基础上 **解耦传感电子与交互表皮**，提供免胶自对齐可更换磁皮肤 + 开源模具/设计工具；**12 s** 级换肤、LSTM 滑移检测 **92%** 准确率，并在 USB 插入等任务上展示 **跨实例零样本策略迁移**（换肤后性能降幅约 **13%**，显著优于 ReSkin **43%** / DIGIT）。

## 开源核查（2026-09-23）

| 项 | 状态 |
|----|------|
| 项目页 | <https://any-skin.github.io/> — 论文、视频、制程说明、Design Tool / 设计文件入口 |
| Python 接口库 | **已开源** — [`github.com/raunaqbhirangi/anyskin`](https://github.com/raunaqbhirangi/anyskin)（`pip install anyskin`；含 Arduino 固件） |
| 夹爪指尖 / 模具设计文件 | **已开源** — [Google Drive 文件夹](https://drive.google.com/drive/folders/1JOb_r0cT0t0BLju4XC6zboPppiXv8LDE?usp=sharing)；项目页 **Design Tool** 按钮链至 GitHub |
| 论文声明 | arXiv 正文写明 **「AnySkin is fully open-sourced」**，代码/设计/视频见项目页 |
| 结论 | **已开源**（设计文件 + 接口代码；非完整 SL 训练栈） |

## 摘要级要点

- **问题：** 触觉传感贵、难集成、实例间信号不一致，换肤后策略常需重训；视觉/本体在文献中占主导，触觉仍是「二等公民」。
- **设计：** 继承 ReSkin 五 magnetometer 电路；**磁化后固化**、**磁弹性体与电路物理分离**、**更细 MQFP-15-7 磁粉** 提升场强与一致性；自对齐磁吸安装（类比手机壳 + 充电线）。
- ** fabrication：** DragonSkin 10 Slow : 磁粉 : 磁粉 = **1:1:2**；脉冲磁化；开源 **两部件模具 CAD 工具** 可从 2D 轮廓生成指尖与模具。
- **实验平台：** xArm、Franka、Leap Hand 等；BC 策略 + 滑移 LSTM（30 类日常物体训练）。
- **跨实例：** 同一策略换肤后 USB 插入等任务仍成功；相对 ReSkin / DIGIT 换肤实验，AnySkin **跨实例泛化** 最优。
- **与旧索引关系：** [`sources/papers/pai_awesome_2401_17695_anyskin.md`](./pai_awesome_2401_17695_anyskin.md) 保留 awesome-physical-ai 策展摘录；**全文级归档以本文件为准**。

## 核心论文摘录（MVP）

### 1) 磁皮肤解耦：可更换、可复用、跨实例一致

- **链接：** <https://arxiv.org/html/2409.08276v1#S3>
- **摘录要点：** 传感电路与易损软表皮分离；平均 **12 s** 更换；新实例无需重标定即可复用已训策略；相对 ReSkin 更均匀磁粉分布与更强磁场。
- **对 wiki 的映射：**
  - [AnySkin（awesome 节点）](../../wiki/entities/painode-146-anyskin.md) — 硬件策展实体；后续可升格独立论文页
  - [触觉传感](../../wiki/concepts/tactile-sensing.md) — 磁触觉 / 可更换表皮设计轴

### 2) 滑移检测与 visuo-tactile 策略学习

- **链接：** <https://arxiv.org/html/2409.08276v1#S5>
- **摘录要点：** 原始 **5×3 轴** 磁通信号可视化；LSTM 滑移分类 **92%**；BC 完成精密插入等接触丰富任务。
- **对 wiki 的映射：**
  - [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)

### 3) 与 DIGIT / ReSkin 的换肤与泛化对照

- **链接：** <https://arxiv.org/html/2409.08276v1#S5.SS3>
- **摘录要点：** 三任务换肤视频；AnySkin 换肤后策略仍成功；ReSkin 换肤性能降幅 **~43%**；AnySkin **~13%**；强调 **首个报告未标定跨实例策略泛化** 的磁触觉皮肤。
- **对 wiki 的映射：**
  - [AnySkin](../../wiki/entities/painode-146-anyskin.md)
  - [OmniTacTune](../../wiki/entities/paper-omnitactune-tactile-residual-adaptation.md) — 触觉表征跨传感器适配对照语境

## 对 wiki 的映射（汇总）

- 策展实体：[AnySkin（painode-146）](../../wiki/entities/painode-146-anyskin.md)
- 概念交叉：[触觉传感](../../wiki/concepts/tactile-sensing.md)、[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)、[接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)
- 项目页归档：[`sources/sites/any-skin-github-io.md`](../sites/any-skin-github-io.md)
- 旧索引（arXiv 已勘误）：[`sources/papers/pai_awesome_2401_17695_anyskin.md`](./pai_awesome_2401_17695_anyskin.md)

## 当前提炼状态

- [x] ICRA 2025 / arXiv:2409.08276 元数据、开源核查、跨实例与滑移结果已摘录
- [x] 与 [`sources/sites/any-skin-github-io.md`](../sites/any-skin-github-io.md) 互证

## BibTeX

```bibtex
@inproceedings{bhirangi2025anyskin,
  title={AnySkin: Plug-and-play Skin Sensing for Robotic Touch},
  author={Bhirangi, Raunaq and Pattabiraman, Venkatesh and Erciyes, Enes and Cao, Yifeng and Hellebrekers, Tess and Pinto, Lerrel},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025},
  eprint={2409.08276},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2409.08276},
}
```
