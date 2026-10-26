# Tactile-Guided Robot Learning: From Representation Learning to Foundation Models（arXiv:2610.20134）

> 来源归档（ingest）

- **标题：** Tactile-Guided Robot Learning: From Representation Learning to Foundation Models
- **类型：** paper / survey / tactile / representation-learning / robot-learning
- **arXiv：** <https://arxiv.org/abs/2610.20134>
- **PDF：** <https://arxiv.org/pdf/2610.20134>
- **配套文献库：** <https://sites.google.com/view/touch-in-robot-learning-review>
- **作者：** Kiyanoush Nazari、Luchen Li、Haoshu Fang、Nima Fazeli、Zhibin Li、Fumiya Iida、Thomas George Thuruthel
- **提交：** 2026-10-14（arXiv v1）
- **入库日期：** 2026-10-26
- **一句话说明：** 从触觉表示、策略学习、基础模型、预测控制与泛化等角度综述触觉如何进入机器人学习与操作。
- **代码 / 数据：** 论文为综述，没有独立可运行代码；作者提供配套文献数据库，收录论文及其项目、代码等资源链接（如有）。

## 摘录 1：触觉表示学习分类（Figure 2）

- **链接：** <https://arxiv.org/abs/2610.20134>
- **摘录要点：** 图 2 将自监督触觉表示学习概括为三类：重建式（Autoencoder、VAE、MAE、隐式场）、对比式（同一传感器增强视图、跨传感器、视触觉跨模态、冻结基础模型锚点），以及非对比/预测式（自蒸馏、潜变量预测、动作条件未来预测）。图示是方法分类示意，不是统一实验结果图。
- **对 wiki 的映射：** [触觉与力觉知识链](../../wiki/overview/hub-tactile.md)、[触觉传感](../../wiki/concepts/tactile-sensing.md)。

## 摘录 2：从表征到动作

- **链接：** <https://arxiv.org/abs/2610.20134>
- **摘录要点：** 综述按触觉在学习与控制管线中的作用组织文献，覆盖触觉表示与策略学习、触觉基础模型、预测与自适应操作，以及跨物体、任务、传感器、具身和仿真到真实的泛化。
- **对 wiki 的映射：** [触觉与力觉知识链](../../wiki/overview/hub-tactile.md)、[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)、[接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)。

## 摘录 3：证据与适用边界

- **链接：** <https://arxiv.org/abs/2610.20134>
- **摘录要点：** 文中以 Sparsh 的 460k+ 无标注触觉观测、六类下游任务作代表性比较，并总结该基准中 DINO 与 I-JEPA 风格潜变量学习的迁移表现强于重建触觉外观的 MAE。该结论来自特定基准，不应外推成所有传感器、任务上的通用排序。
- **对 wiki 的映射：** [Sparsh](../../wiki/entities/paper-sparsh.md)、[触觉与力觉知识链](../../wiki/overview/hub-tactile.md)。

## 当前提炼状态

- [x] 论文标题、作者、提交日期和 Figure 2 图注已核对
- [x] 配套文献数据库已核对；论文自身没有独立运行代码
- [x] 与触觉知识链、已有触觉综述及 Sparsh 基准建立交叉引用

## BibTeX

```bibtex
@misc{nazari2026tactileguidedrobotlearningrepresentation,
  title={Tactile-Guided Robot Learning: From Representation Learning to Foundation Models},
  author={Nazari, Kiyanoush and Li, Luchen and Fang, Haoshu and Fazeli, Nima and Li, Zhibin and Iida, Fumiya and Thuruthel, Thomas George},
  year={2026},
  eprint={2610.20134},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2610.20134},
}
```
