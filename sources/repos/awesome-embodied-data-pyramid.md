# Awesome Embodied Data Pyramid（worldbench/awesome-embodied-data-pyramid）

> 来源归档

- **标题：** Awesome Embodied Data Pyramid
- **类型：** repo / awesome-list / dataset-curation
- **来源：** Data Pyramid for Embodied Manipulation 综述团队（PKU 牵头 11 机构；`worldbench` 组织）
- **链接：** <https://github.com/worldbench/awesome-embodied-data-pyramid>
- **项目页：** <https://jasper-aaa.github.io/embodied-data-pyramid/> — 见 [`sources/sites/embodied-data-pyramid.md`](../sites/embodied-data-pyramid.md)
- **论文：** <https://arxiv.org/abs/2607.24744>
- **许可：** Apache-2.0（仓内 `LICENSE` 为 Apache License 2.0；内容为数据集与论文策展索引）
- **入库日期：** 2026-07-29（2026-11-17 复查 11-16 工程发布）
- **核对范围：** 2026-11-17 重新核对官方 README 的五层目录、News 与 DataPyramid 发布声明；逐页阅读 tools/lerobot_pipeline/README.md、docs/QUICKSTART.md、VALIDATION.md、docs/RELEASE_VALIDATION.md，并审阅 pipeline.py、pipeline/cli.py、pipeline/conversion.py、pipeline/release.py 和 quality/src/lerobot_quality/{export,batch_export,four_rules}.py 的运行入口及主要函数；原 2026-11-14「无 Python 脚本」结论因 11-16 更新失效。
- **通读：** 已完成
- **一句话说明：** 综述配套的开源策展仓：持续维护五层数据与 VLA/WAM 配方清单；2026-11-16 新增 DataPyramid 六类数据统一转换为 LeRobot v3、按帧质量清洗的可运行 Python 管线与 ModelScope 数据入口。
- **沉淀到 wiki：** [`wiki/entities/paper-data-pyramid-embodied-manipulation.md`](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md)

---

## 开源边界（步骤 2.5，2026-11-17 复核）

| 公开产物 | 开放状态 / 边界 |
|----------|-----------------|
| 五层数据集 / VLA-WAM 策展清单（`README.md`） | 已发布；资源导航，不等于引用的数据集均可下载 |
| [LeRobot v3 处理管线](https://github.com/worldbench/awesome-embodied-data-pyramid/tree/main/tools/lerobot_pipeline) | **已发布可运行代码**：六类转换器、统一 CLI、逐帧质量导出、测试与验证记录；需另备原始数据、运行环境和获授权的 FK 模型 |
| [DataPyramid ModelScope 集合](https://www.modelscope.cn/datasets/jasper0519/DataPyramid) | 正在上传（官方 2026-11-16 声明）；**清单处理规模不等于当日可下载量** |
| 模型训练/推理实现与权重 | 本次发布是数据工程管线，不是具身模型训练或推理仓库 |

## DataPyramid 数据工程发布（2026-11-16）

官方处理清单覆盖 **AgiBot Beta、ABC-130k、AgiBotWorld2026、Galaxea、InternData A1、RoboMIND** 六个既有家族，转换为 **LeRobot v3**。官方盘点为 **1,721 仓库、972,324 episodes、1,077,451,809 帧、10,207.57 轨迹小时**（包括实机与仿真），并非独立新采集的数据集。

- **运行入口：** `tools/lerobot_pipeline/pipeline.py` → `pipeline.cli.main`；`doctor` 检查依赖、`plan` 生成工作计划；`run --execute` 由 `pipeline.conversion.build_steps` 规划家族脚本，并经 `pipeline.release.verify` 核查 exact manifest 的路径、Schema、轨迹与帧数。
- **清洗入口：** `lerobot_quality.batch_export` 调用 `lerobot_quality.export.export_repo`；基础数值/索引/mask 检查加 alpha10 四项规则（范围离群、速度/加速度突变、FK 一致性、状态–动作一致性）。未适用的检查记录跳过原因。
- **训练接口：** 所有原始帧、列与时间索引保留，仅增加逐帧布尔列 `quality.valid`。训练必须显式使用该标记和已有 action/EE masks；视频软链接依赖已转换数据，`quality/` 是诊断报告。
- **版本与许可：** ModelScope 正在上传的是 **20261016** 质量注释；公开代码同时提供较新的 **alpha10** 配置，不能视为线上 Parquet 已替换。FK 模型需要单独授权引入，源数据许可证独立适用。
- **验证边界：** [官方 2026-11-16 验证记录](https://github.com/worldbench/awesome-embodied-data-pyramid/blob/main/tools/lerobot_pipeline/docs/RELEASE_VALIDATION.md) 报告既有全量处理结果的清单校验及回归测试；**本次 ingest 未独立重跑**六类全量转换/清洗，官方未声称全新环境安装或逐帧解码全部视频。

一手资料：[官方仓库](https://github.com/worldbench/awesome-embodied-data-pyramid) · [处理管线说明](https://github.com/worldbench/awesome-embodied-data-pyramid/blob/main/tools/lerobot_pipeline/README.md) · [QUICKSTART](https://github.com/worldbench/awesome-embodied-data-pyramid/blob/main/tools/lerobot_pipeline/docs/QUICKSTART.md)。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-data-pyramid-embodied-manipulation](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md) | 综述实体页（五层 taxonomy、模型数据配方、六大挑战） |
| [awesome-wam-openmoss](awesome-wam-openmoss.md) | 同类 Awesome 策展范式（WAM 论文库）；本清单按「数据层」而非「模型族」组织 |
| [open-x-embodiment](../../wiki/concepts/open-x-embodiment.md) | 真机层聚合数据集代表，清单收录对象之一 |
