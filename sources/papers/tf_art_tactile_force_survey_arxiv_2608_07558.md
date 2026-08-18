# Learning Physical Interaction: A Survey of Tactile- and Force-aware Robot Learning（arXiv:2608.07558）

> 来源归档（ingest）

- **标题：** Learning Physical Interaction: A Survey of Tactile- and Force-aware Robot Learning
- **缩写 / 框架：** **TF-ART**（Tactile/Force-Aware Robot learning Taxonomy）
- **类型：** paper / survey / tactile / force-aware / contact-rich
- **arXiv：** <https://arxiv.org/abs/2608.07558>
- **PDF：** <https://arxiv.org/pdf/2608.07558>
- **项目页：** <https://lorenzo-0-0.github.io/tactile-force-survey/>（归档见 [`sources/sites/lorenzo-tactile-force-survey.md`](../sites/lorenzo-tactile-force-survey.md)）
- **代码 / 清单：** <https://github.com/NTUMARS/Awesome-Tactile-Force-aware-Robot-Learning>（归档见 [`sources/repos/awesome-tactile-force-aware-robot-learning.md`](../repos/awesome-tactile-force-aware-robot-learning.md)）
- **作者：** Shilin Shan、Chuhao Zhou、Ruize Wang、Xinyan Chen 等（NTU 通讯 Jianfei Yang）
- **机构：** 南洋理工大学（NTU）；斯坦福；UC Berkeley；MIT；NUS；Georgia Tech；东京大学；ETH Zurich；Harvard；Imperial；KTH；TU Darmstadt
- **入库日期：** 2026-08-18
- **一句话说明：** 用统一层级把触觉/力觉机器人学习同时映射到多模态观测与多阶段策略–控制管线；配套 Awesome 清单按主模型结构分组。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-18）：** [lorenzo-0-0.github.io/tactile-force-survey](https://lorenzo-0-0.github.io/tactile-force-survey/) 自称覆盖 **266** 篇参考文献的交互式 TF-ART 框架，链到 GitHub 清单。
- **代码仓：** [NTUMARS/Awesome-Tactile-Force-aware-Robot-Learning](https://github.com/NTUMARS/Awesome-Tactile-Force-aware-Robot-Learning) 仅 `README.md` 策展列表（按 VLA / RL / DP / ACT / 回归 / 混合分组），**无可运行训练/评测入口**。
- **结论：** **已开源（Awesome 清单 + 项目页）**；源码运行时序图不适用。

## 摘录 1：问题

接触敏感操作不只靠看见和出动作，还要 **调力与自适应控制**。已有综述多按传感器、任务或学习范式切，很少同时覆盖多模态融合与「高层策略 → 动作细化 → 底层力/柔顺控制」全管线。

## 摘录 2：TF-ART

层级：观测空间 → 编码/融合 →（可选）输入重建 → 主动作生成 →（可选）缺失模态预测 / 中间量 → 动作细化 → 机器人端力控。每篇方法只按其 **主模型结构** 出现一次，多模态与多阶段属性写在条目下。

## 摘录 3：与站内关系

对照 [Awesome Touch](../../wiki/entities/awesome-touch.md)（2025–2026 VTLA/WAM 精选）：TF-ART 是 **管线轴 taxonomy + 更宽时间窗**，不是同一清单的镜像。

**对 wiki 的映射：** [`wiki/entities/paper-tf-art-tactile-force-survey.md`](../../wiki/entities/paper-tf-art-tactile-force-survey.md)；交叉 [Tactile Sensing](../../wiki/concepts/tactile-sensing.md)、[Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)、[触觉知识链](../../wiki/overview/hub-tactile.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（清单已开源、无可运行训练）
