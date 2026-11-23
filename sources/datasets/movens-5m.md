# Movens-5M：多模态、执行验证的人形运动数据集

> 来源归档（dataset；非另立论文实体）

- **标题：** Movens-5M — Large-scale, multimodal, execution-verified humanoid motion data from Internet videos
- **类型：** dataset
- **官方数据卡：** https://huggingface.co/datasets/wendell0218/Movens-5M
- **论文：** https://arxiv.org/html/2610.41877 （arXiv:2610.41877，v1；全文含附录 A–G）
- **项目：** https://movens-zero.github.io/
- **源码：** https://github.com/wendell0218/Movens-Zero
- **核对日期：** 2026-11-23
- **核对范围：** 官方 Hugging Face Dataset Card 的 Overview、Full dataset access、Data format、Data construction pipeline、Data loading、Limitations 与 License；结合论文第 3、5–6 节及附录 A–B 校核物理验证口径。
- **通读：** 已完成
- **沉淀到 wiki：** [Movens-Zero / Movens-5M](../../wiki/entities/paper-movens-zero.md)

## 数据口径与来源

- 论文及官方数据卡报告 **1,500 万条人体运动序列、500 万条物理可执行的人形运动序列、超过 10,000 小时、297 类动作、16 类场景**，以及文本、RGB 视频、语音、音乐四种条件。这里的 500 万条属于研究数据集统计，**不是这个中央 HF 仓库的单仓行数**。
- 数据由互联网视频经片段切分、视觉与运动质量筛选、语义标注、人体运动恢复、物理修复及人形重定向、闭环跟踪验证得到。
- 在官方卡里，每条预览记录以 `sample_id`/`sample_name` 关联 `original_video`、`rendering_video`、`retargeted_video`、`tracking_video`，以及 `motion_type`、`motion_caption`、`video_caption`、`duration`、`fps` 等字段。年龄/性别/身高等属于自动注释，不能当作人工核实真值。

## 实际公开入口（务必区分）

中央入口 [wendell0218/Movens-5M](https://huggingface.co/datasets/wendell0218/Movens-5M) 发布 **Parquet 预览**（该页面显示存储文件约 5.52 GB）；官方数据卡说明完整发布在下面四个以 tar shard 组织的仓库：

1. [Movens-0/Movens-5M-0](https://huggingface.co/datasets/Movens-0/Movens-5M-0)
2. [Movens-2/Movens-5M-1](https://huggingface.co/datasets/Movens-2/Movens-5M-1)
3. [Movens-1/Movens-5M-2](https://huggingface.co/datasets/Movens-1/Movens-5M-2)
4. [Movens-1/Movens-5M-3](https://huggingface.co/datasets/Movens-1/Movens-5M-3)

**不要把「四个 tar 仓库已列出」写成「已经下载并完整校验 500 万条」。** 本次核对了官方发布说明，但没有实际下载全量 shards、去重和逐样本运行验证。中央预览的 Dataset Viewer 可能报错；也不等同于原始文件不可下载。

## 许可、复现与边界

- 中央 Hugging Face 页面 License 标注为 **`other`**；不应推断为 CC-BY、MIT 或可自由商用。各分片和视频原始版权/肖像隐私/用途约束需按对应发布条款另行确认。
- 论文中的「物理可执行」是物理修复、重定向和闭环 tracking 产生的筛选结论，不等同于所有轨迹已在真机逐条安全复现。
- Movens-Intent 是**另一套评测集**（8,192 条专家核验条件–运动配对、四模态各 2,048 条），不是 Movens-5M 同义词：[Movens-Intent 官方入口](https://huggingface.co/datasets/wendell0218/Movens-Intent)。
- 用于训练前应先抽检 sample_id、fps、qpos 顺序、音视频同步、场景/动作分布、许可和物理跟踪成功率。

## 对 wiki 的映射

- [Movens-Zero / Movens-5M：数据→模型→控制完整链路](../../wiki/entities/paper-movens-zero.md)
- [具身数据纵深路线：Stage 4](../../roadmap/depth-embodied-data.md)
