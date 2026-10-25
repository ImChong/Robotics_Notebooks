# MorphMimic

> 来源归档

- **标题：** MorphMimic: Morphology-Conditioned World Model and Policy Sharing Across Humanoids
- **类型：** repo（论文作者维护的实现）
- **链接：** <https://github.com/lzyang2000/morphmimic>
- **项目页：** <https://lzyang2000.github.io/morphmimic/>（[来源归档](../sites/morphmimic-project.md)）
- **技术报告：** <https://lzyang2000.github.io/morphmimic/static/morphmimic.pdf>（[来源归档](../papers/morphmimic_technical_report_2026.md)）
- **作者：** Lizhi Yang、Fengze Xie、Yuyang Wang、Lu Gan、Yisong Yue、Aaron D. Ames
- **机构：** 加州理工学院、佐治亚理工学院
- **许可证：** Apache-2.0（仓库代码；模型文件和第三方依赖保留各自许可证）
- **入库日期：** 2026-10-25
- **再核日期：** 2026-10-25
- **一句话说明：** MorphMimic 的官方代码实现，提供 BONES-SEED 数据转换与筛选、多机器人动作重定向、可微世界模型和策略训练、ONNX 导出、MuJoCo 仿真以及 Unitree G1 部署入口。
- **沉淀到 wiki：** [MorphMimic 研究实体](../../wiki/entities/paper-morphmimic.md)

---

## 开源状态（2026-10-25）

| 项 | 状态 |
|---|---|
| 训练与评测代码 | **已开源**；src/morphmimic/ 含训练器、环境、策略、critic、世界模型及评测相关模块 |
| 机器人 | Unitree G1、AgiBot X2 Ultra、Booster T2；通过规范化关节映射与形态描述符共用网络 |
| 数据 | 训练流程使用 BONES-SEED；需自行准备对应数据文件。X2 Ultra 与 T2 动作由 SOMA Retargeter 从共同的人体动作源重定向 |
| 部署 | 支持训练 checkpoint 导出 ONNX；仓库提供 MuJoCo CPU Sim2Sim、PICO VR 五点遥操作，以及 Unitree G1 硬件节点 |
| 关系 | README 说明其 SuperMimic 基础算法是作者独立实现，并非 SuperMimic 作者代码 |

## 复现线索

README 给出的训练默认项包括 4096 个并行环境/每 GPU、32 步想象 horizon、40,000 轮训练；三个机器人联合训练需要一机三 GPU。仓库说明 Python 3.11、mjlab 1.6.0、Torch CUDA 12.8 环境。部署时策略可导出为一个跨机器人使用的 ONNX 模型，并为每台机器人单独保存归一化统计、形态描述符和常量配置。

## 对 wiki 的映射

- [MorphMimic 研究实体](../../wiki/entities/paper-morphmimic.md) — 报告、项目页和官方代码的统一节点
- [MorphMimic 技术报告](../papers/morphmimic_technical_report_2026.md)
- [MorphMimic 官方项目页](../sites/morphmimic-project.md)
