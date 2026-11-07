# RoboSteer 项目页

> 来源归档

- **标题：** RoboSteer 项目页
- **类型：** site
- **链接：** <https://robosteer.github.io/>
- **论文：** <https://arxiv.org/abs/2610.10198>（见 [robosteer_arxiv_2610_10198.md](../papers/robosteer_arxiv_2610_10198.md)）
- **代码：** 截至 2026-11-07 未列可运行的完整官方模型适配/评测源码仓库；提供托管式网页评测入口不能视为源码开源
- **数据集：** <https://huggingface.co/datasets/YanCORANV/RoboSteer>（已发布，见 [repos/robosteer.md](../repos/robosteer.md)）
- **入库日期：** 2026-10-12
- **最近核对：** 2026-11-07
- **一句话说明：** 行为可操控性基准页，列出 Hugging Face 数据、评测器权重和模型资源；另有托管式在线评测入口，但完整官方适配及评测源码仍未公开。

## 页面列出的资源（2026-11-07 复核）

| 资源 | 链接 | 状态 |
|------|------|------|
| 基准数据集 | huggingface.co/datasets/YanCORANV/RoboSteer | 已发布 |
| 评测器权重 | huggingface.co/Leontc/RoboSteer-evaluator | 已发布（含权重与指标 registry） |
| 评测数据 | huggingface.co/datasets/PhoebeCC/RoboSteer_Evaluation | 已发布 |
| 模型资源 | huggingface.co/datasets/PhoebeCC/RoboSteer-Model-Resources | 仅文档/manifest，「可运行模型包尚未发布」 |
| 中间数据 | huggingface.co/datasets/PhoebeCC/RoboSteer-Preprocessing | 已发布（Level 1 处理后文本与图像条件静态视频） |

## 开源核查结论

部分开放：数据与评测器权重已公开，项目页提供托管式在线评测入口；完整的官方模型适配与评测代码仓库和可运行模型包未公开，不能凭在线评测服务推断代码调用路径。

## 沉淀到 wiki

- [`wiki/entities/paper-robosteer-behavior-steerability-benchmark.md`](../../wiki/entities/paper-robosteer-behavior-steerability-benchmark.md)
