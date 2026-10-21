# WorldOlympiad: Can Your World Model Survive a Triathlon?

> 一手论文摘录；与官方项目页和代码仓库配套。

- **类型：** paper
- **作者 / 机构：** Yuke Zhao 等；浙江大学、阿里巴巴达摩院、香港科技大学、Monash University、TRE
- **论文：** <https://arxiv.org/abs/2606.11129>
- **HTML：** <https://arxiv.org/html/2606.11129v2>
- **项目页：** <https://alibaba-damo-academy.github.io/WorldOlympiad/>
- **代码：** <https://github.com/alibaba-damo-academy/WorldOlympiad>
- **数据集：** <https://huggingface.co/datasets/ziplab/WorldOlympiad>
- **入库日期：** 2026-10-16
- **一句话说明：** 以物理规律、3D 几何一致性和长程交互保真度三项诊断视频世界模型；基准包含 400 段机器人操作、400 段游戏和 200 段真实世界长视频。

## 核心摘录

### 三项诊断维度

- **Physical track：** 以物体分割和多模态大模型裁判，考察力学、热现象与材料属性是否符合可解释规律。
- **Geometry track：** 使用 Depth Anything 3 和 Gaussian splatting 重建生成视频，诊断结构一致性、新视角质量和相机轨迹对齐。
- **Interaction track：** 将长视频按连续片段切分，检查片段级指令遵循、相邻片段过渡、全视频一致性与语义对齐。

### 数据与评测范围

官方项目页列出共 **1,000** 段评测视频：机器人部分来自 RoboCOIN 双臂操作视频，游戏部分采样自 OGameData_50K，真实世界部分采样自 LVD-2M 长视频。三类数据共同用于测试跨领域视频世界建模能力，不应把机器人子集的结果解释成真实机器人策略成功率。

### 结果如何解读

项目页显示自动排名与人工偏好排名的 Spearman 相关为 **0.95**；同时分别展示物理、几何、语义与交互结果。使用时应保留分项诊断，避免只看 Overall 分数而忽略长时序 3D 一致性等薄弱项。榜单结果是项目页当前公布的模型与协议结果，不等于所有生成模型的永久排名。

## 对 wiki 的映射

- [WorldOlympiad](../../wiki/entities/worldolympiad.md) — 基准定位、数据域、评测维度与限制。
- [具身大模型评测基准选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) — 作为开放域长视频物理/几何/交互诊断基准，与操作视频和策略效用基准区分。

## 一手来源

- [论文](https://arxiv.org/abs/2606.11129)
- [项目页](https://alibaba-damo-academy.github.io/WorldOlympiad/)
- [官方代码](https://github.com/alibaba-damo-academy/WorldOlympiad)
- [Hugging Face 数据集](https://huggingface.co/datasets/ziplab/WorldOlympiad)
