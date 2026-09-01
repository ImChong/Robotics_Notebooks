# OptiGeo（光学挑战场景单目几何）

> 来源归档（ingest）

- **标题：** OptiGeo: Efficient Monocular Geometry for Embodied Perception in Optically Challenging Scenes
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.29881>
- **机构：** 香港大学（HKU）等
- **项目页：** <https://mx-liu6.github.io/OptiGeo-web/>
- **代码：** <https://github.com/mx-liu6/OptiGeo>
- **模型：** <https://huggingface.co/mxliu-hku/OptiGeo>
- **入库日期：** 2026-09-01
- **一句话说明：** 将透明/反光/镜面场景深度失真重定义为基础模型训练中的 sensor-induced supervision bias；bias-aware training + clean-geometry teacher；30M 参数超更大单目与多视角基线。

## 核心摘录（MVP）

### 1) 问题：光学挑战场景监督偏差

- **摘录要点：** 单目深度在透明/反光/镜面环境不可靠；现有方法常靠场景特定预处理或事后微调，易冗余与过拟合。
- **对 wiki 的映射：**
  - [OptiGeo](../../wiki/entities/paper-optigeo.md)

### 2) bias-aware training

- **摘录要点：** clean-geometry teacher + residual-trimmed alignment 修复有偏真实监督；透明目标渲染作紧凑干净光学几何来源。
- **对 wiki 的映射：**
  - [OptiGeo](../../wiki/entities/paper-optigeo.md)

### 3) 开源状态（截至 2026-09-01）

- **摘录要点：** **已开源** `mx-liu6/OptiGeo` + Hugging Face 权重。
- **对 wiki 的映射：**
  - [OptiGeo 项目页](../sites/optigeo.md)
  - [mx-liu6/OptiGeo](../repos/mx-liu6-optigeo.md)

## 当前提炼状态

- [x] 项目页/仓库核查
- [x] wiki 映射：`wiki/entities/paper-optigeo.md` 新建
