# Fiatlux 遥操作数据

- **类型：** dataset
- **URL：** <https://huggingface.co/datasets/haw-ai-i/fiatlux-teleoperation>
- **项目页：** [Fiatlux](../sites/fiatlux.md)
- **论文：** [Fiatlux 论文归档](../papers/fiatlux_arxiv_2609_38216.md)
- **代码：** [Fiatlux 仓库归档](../repos/fiatlux.md)
- **入库 / 数据卡核查日期：** 2026-10-02
- **许可：** CC-BY-4.0
- **实体页：** [Fiatlux](../../wiki/entities/paper-fiatlux.md)

## 数据范围

HF 数据卡记录：125 episodes、约 3.7 GB。八个非攀爬子任务含 80 个满分成功与 27 个失败 takes；另有 18 个攀爬尝试。**论文/项目页的 107 个 takes 仅统计非攀爬部分**，不能据此认定数据卡矛盾，也不能把攀爬尝试视为成功示范。

每条记录含 `run.h5`、`meta.json`、评分报告与第一/第三人称视频。`meta.json` 提供关节顺序、seed、阈值、版本等；使用仓库 `score.py` / `score_subtasks.py` 复读。数据卡提示历史在 2026-09-21 被展平，部分记录的旧 commit SHA 已无法解析，复现时保存版本与配置。

数据来自 Isaac Sim 中的 G1 + Dex3 仿真遥操作；不含真机记录，也不含模型权重。与项目页其他手型/自由度配置区分。
