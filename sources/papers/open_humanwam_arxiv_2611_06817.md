# Open-HumanWAM：规模化人类与机器人经验的开放配方 / An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models

> 来源归档（paper）；以下为**一手原文阅读摘录与数值口径核对**，不是原文全文转载。

- **标题：** Open-HumanWAM: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.06817>
- **原文版本：** arXiv:2611.06817v2（2026-11-05）；首次提交 2026-11-04
- **全文（含附录）：** <https://arxiv.org/html/2611.06817>
- **PDF：** <https://arxiv.org/pdf/2611.06817>
- **项目页：** <https://dagroup-pku.github.io/open-humanwam/>
- **代码：** <https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM>
- **机构：** 北京大学、新加坡国立大学、麻省理工学院、加州大学圣巴巴拉分校、NVIDIA
- **入库日期：** 2026-11-06；2026-11-08 补充原文通读；2026-11-09 核对官方数据卡与代码分支
- **核对范围：** arXiv v2 HTML 全文：引言、数据与统一动作、Vanilla/Goal 方法、缩放与消融、RoboTwin/RoboDojo、真机实验、限制讨论、附录评测口径及实现细节；另比对官方仓库总 README 与两条模型 README
- **通读：** 已完成
- **一句话说明：** 50K 小时人类第一人称动作语料支持纯人类预训练与人机联合目标条件预训练两条路径；应严格区分缩放、零样本与最终后训练的实验。

## 核心内容（按原文实验逻辑整理）

1. **语料（数据章节）：** HumanNet、EgoSuite-Open100K、EgoDex、Gen-HumanEgo、EgoVerse 汇集约 50K 小时与约 35.9M clips；先筛选手部可见性、运动稳定与非静止片段，再统一摄像机/腕部与左右臂 EEF 表示，每臂 3D 位置+4D 四元数+1D 夹爪，合计 16D。
2. **HumanWAM-Vanilla（方法及缩放章节）：** 从 Wan2.2-TI2V-5B 加 action expert，考察具有未来视觉条件的 IDM attention 与其它注意力设计；在 EgoPro 中按 W0 进行多视角中训练，再用机器人演示后训练。受控 IDM 扩展 100→50,000 小时，Scene-OOD 49.1%→73.2%，体现测试范围内的经验缩放关系。
3. **HumanWAM-Goal（方法章节）：** 基于 Cosmos3-Nano-16B 的视觉目标条件 world-action model；输入文本、图像/视觉历史、目标图像和机器人状态，输出视频/动作预测；固定 14,618 小时机器人数据，增加最多 45,938 小时人类数据研究可迁移性。未来视频与动作的联合建模不代表纯视频生成可以替代动作约束。
4. **控制变量（实验章节）：** 在固定 robot-only 基线下，增加人类数据使 RoboTwin 零样本成功率 11.48%→13.94%，20 个真机任务平均零样本 20.0%→30.5%；受控后训练的 RoboTwin Random 成功率 70.80%→74.36%。三组测量目的、后训练状态与评测协议不同。
5. **最终模型（结果章节/附录）：** Vanilla RoboTwin Clean2Clean 92.08%、Clean2Random 68.88%、均值 80.48%；RoboDojo 成功率 14.17%、进度 19.00。Goal 最终 RoboTwin Clean2Clean 92.70%、Clean2Random 84.60%、均值 88.65%（50 项任务、100 官方种子、Clean 与 Random，计 10,000 episodes）。
6. **限制与复现（讨论/附录）：** 规模关系不应外推到任意时长；动作表示/任务协议与数据清洗影响复现，真机目标视图、相机布局及指令仍对结果有影响；Goal 的任务后训练不能与未更新目标任务参数的零样本实验混写。

## 重要数字口径

| 数字 | 具体含义 | 不可混同 |
|---|---|---|
| 50,000 h | Open-HumanWAM-50K 人类视频语料规模 | 不是 Goal 混合训练实际使用的 45,938 h 人类子集 |
| 14,618 h | Goal 数据消融中固定的机器人预训练经验 | 不是训练增量 |
| 49.1%→73.2% | Vanilla IDM 的 Scene-OOD 随人类预训练扩展的受控结果 | 不是 RoboTwin 最终模型的 Random 成功率 |
| 70.80%→74.36% | Goal 受控人类数据增量后训练的 RoboTwin Random | 不是最终 Goal 的 84.60% |
| 92.70%/84.60% | Goal 最终 Clean/Random | 属于后训练评测，非零样本成功率 |
| 50 vs 100 | 当前 Vanilla README 脚本用每任务每条件 50 episodes；论文结果用 100 | 公开 checkpoint 本地回归与论文表格不能无标注等同比对 |

## 50K 数据集当前发布口径（2026-11-09）

- **数据卡：** <https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K>；LeRobot v3，31,396,799 实体 episodes（不同于论文统计的 35.9M 选中 clips）。
- **2026-11-09 14:53 UTC+8 快照：** 上传并验证 22.43/26.67 TB，84.09%；完整 payload 仍在上传。HF 访问需接受条款。七档训练 subset 索引已可见，数据 payload 需另行取得。
- **复现注意：** 数据来源分属不同授权条款；不能由顶层代码 Apache-2.0 推导视频数据许可。

## 开源状态

- **部分开源（2026-11-08 静态核对）**：仓库给出 Vanilla 训练/评测与 Goal 推理代码，部分 HF 权重可下载；50K 全量语料仍标为上传中，Goal 机器人后训练脚本未发布。开放程度以官方仓库 README / 模型卡后续更新为准。
- **许可证**：仓库顶层 Apache-2.0；Goal 的 Cosmos 衍生实现另有 OpenMDW-1.1 组件条款，数据/权重许可另核。
- **验证范围**：已读论文 HTML 正文与附录及相关源码静态入口；**未独立训练或跑 GPU benchmark**。

## 对 wiki 的映射

- [Open-HumanWAM 唯一实体节点](../../wiki/entities/paper-open-humanwam.md)
- [具身智能小站 11-06 阅读坐标](../../wiki/overview/embodied-station-2026-11-06-20-papers-technology-map.md)
