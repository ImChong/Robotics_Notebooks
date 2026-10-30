# 四项多模态基准：一手资料与待核查记录

- **类型：** site / benchmark / leaderboard
- **记录日期：** 2026-10-30
- **状态：** 收录草稿；本次尚未通读四篇论文及核查评测源码，不应视为完成 ingest。

## 原始入口

| 基准 | 测量对象 | 一手资料 | 本库已有节点 |
|------|----------|----------|--------------|
| Daily-Omni | 音视频时序对齐与推理 | [项目页及榜单](https://lliar-liar.github.io/Daily-Omni/#leaderboard) · [论文](https://arxiv.org/abs/2505.17862) · [代码](https://github.com/Lliar-liar/Daily-Omni) · [数据](https://huggingface.co/datasets/liarliar/Daily-Omni) | [Daily-Omni](../../wiki/entities/paper-daily-omni.md) |
| TimeLens-Bench | 视频时序定位 | [项目页及榜单](https://timelens-arc-lab.github.io/#leaderboard) · [论文](https://arxiv.org/abs/2512.14698) · [代码](https://github.com/TencentARC/TimeLens) · [数据](https://huggingface.co/datasets/TencentARC/TimeLens-Bench) | 待深入检索 |
| EmbSpatial-Bench | 具身第一视角空间关系理解 | [ACL 论文](https://aclanthology.org/2024.acl-short.33/) · [代码与数据入口](https://github.com/mengfeidu/EmbSpatial-Bench) | [EmbSpatial](../../wiki/entities/embspatial.md) |
| MMAU | 音频理解与推理 | [项目页及榜单](https://sakshi113.github.io/mmau_homepage/) · [论文](https://arxiv.org/abs/2410.19168) · [代码](https://github.com/Sakshi113/mmau) | 待深入检索 |

## 2026-10-30 可核对的结果

- Daily-Omni 官方榜单的音视频平均准确率：AGIBOT WITA-Omni 1.0 为 86.22%，Preview 为 85.21%；这是两个不同版本。官方说明部分比较模型的有效样本数存在差异，需对照榜单脚注。
- TimeLens 官方仓库 2026-10-26 新闻记载 WITA-Omni 1.0 的 TimeLens-Bench 平均分为 66.4。完整名次和比较设置需在官方榜单核实。
- EmbSpatial-Bench 和 MMAU：尚未从基准作者的公开结果中核实 WITA-Omni 的成绩、数据划分和名次；不能将活动图表直接当作官方榜单成绩。
- MMAU 官方页面区分 MMAU-v05.15.25、旧版及未经团队验证的社区结果；比较时必须说明版本和 test / test-mini。

## 后续完成条件

1. 通读四篇论文全文，核对版本、数据构建、指标、实验与局限。
2. 核查各官方仓库实际运行入口和评测代码，并在对应实体页补充基于原文的项目流程图及基于源码的运行时序图。
3. 更新已有 Daily-Omni、EmbSpatial 节点；确认身份后再为未收录的基准建实体并补充来源与交叉引用。
4. 补全结果溯源，运行仓库规定的 `make ci-preflight` 和相关检查，通过后再将 PR 转为待审。
