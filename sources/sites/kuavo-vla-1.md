# KUAVO-VLA-1.0（乐聚官方项目页）

> 来源归档（site）

- **标题：** KUAVO-VLA-1.0
- **类型：** 项目页 / 垂域视觉-语言-动作模型
- **官方项目页：** <https://model.lejurobot.com/kuavo-vla-1/>
- **代码：** <https://github.com/LejuRobotics/LeTools-Learning>
- **模型卡：** <https://huggingface.co/LejuRobotics/LET-KUAVO-VLA-1.0-models> · <https://www.modelscope.cn/models/lejurobot/LET-KUAVO-VLA-1.0-models>
- **数据集卡：** <https://huggingface.co/datasets/LejuRobotics/LET-KUAVO-VLA-1.0-Dataset> · <https://www.modelscope.cn/datasets/lejurobot/LET-KUAVO-VLA-1.0-Dataset>
- **社区入口：** <https://openlet.openatom.tech/>
- **来源核查日期：** 2026-10-03
- **一句话说明：** 乐聚面向 Kuavo 本体与工业场景发布的垂域 VLA；配套代码位于 LeTools-Learning，模型与数据通过 OpenLET 申请并经审核获取。

## 项目与数据状态

| 资源 | 当前可见状态 | 备注 |
|------|-------------|------|
| LeTools-Learning 代码 | 已公开，仓库许可证为 GPL-3.0 | 包含 LeRobot 数据转换、策略训练、仿真/真机部署工具；代码许可不等于模型权重许可 |
| KUAVO-VLA-1.0 模型 | Hugging Face 页面要求同意分享联系信息并通过访问审核 | 模型卡称仓库包含权重和配置；获批后的模型许可应以页面条款为准 |
| 配套数据集 | OpenLET 标为申请审核；HF 数据卡列出 Apache-2.0 元数据并要求审核 | 审核门槛与许可证是两件事，使用前仍需确认适用范围和署名义务 |
| ModelScope 镜像 | 项目发布截图列出模型与数据集镜像入口 | 以平台当前页面与访问政策为准 |

## 截图所载发布信息（厂商口径）

用户提供的项目发布截图称：

- 基于同构型 Kuavo 真机数据进行 **600+ 小时**二次训练，覆盖分拣、搬运、上下料、装配等 **100 项**工业典型任务。
- 在 **25 项任务 Benchmark** 上综合成功率为 **48.27%**、过程得分为 **74.51%**，并称相较通用基模提升超过 30 个百分点。
- 新任务开发声称算力与数据成本下降 **50%+**、训练周期从数周压缩到数天。

这些数字来自截图中的发布文案；当前可核对的 GitHub README 和模型卡未提供相同评测协议、基线或原始结果表，因此应视为厂商报告值，不能据此直接与其他 Benchmark 横向比较。

## 对 wiki 的映射

- 项目知识页：[KUAVO-VLA-1.0](../../wiki/entities/kuavo-vla-1.md)
- 通用训练与部署栈：[LeTools](../../wiki/entities/letools.md)
- 代码归档：[letools-learning.md](../repos/letools-learning.md)
- 数据社区：[OpenLET](../../wiki/entities/openlet.md)

## 核验入口

- [LeTools-Learning README](https://github.com/LejuRobotics/LeTools-Learning) 将 KUAVO-VLA-1.0 列为基于 Kuavo 与工业场景的垂域模型，并介绍 LeRobot 转换、训练与部署流程。
- [Hugging Face 模型卡](https://huggingface.co/LejuRobotics/LET-KUAVO-VLA-1.0-models) 指向项目页与代码仓，并说明访问需审批。
- [OpenLET 社区](https://openlet.openatom.tech/) 说明模型及配套数据开放申请，审核后可在个人中心查看仓库。
