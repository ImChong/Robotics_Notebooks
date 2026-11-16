# Open-HumanWAM-50K（人类第一视角动作经验语料）

> 来源归档（dataset），依据 **DAGroup-PKU 官方 Hugging Face 数据卡**，核查日期 2026-11-11。上传数字为时间戳快照，不代表此刻实时进度。

- **类型：** 自我中心（egocentric）视频 + 手部动作 + 自然语言标注数据集
- **数据卡：** <https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K>
- **论文：** <https://arxiv.org/abs/2611.06817>
- **项目：** <https://dagroup-pku.github.io/open-humanwam/>
- **代码：** <https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM>
- **提供方：** DAGroup-PKU
- **核对范围：** Hugging Face 数据卡的访问要求、上传状态、格式、来源、训练 subset、版权条款说明；官方仓库 README 的 Corpus 及 Release Plan。
- **一句话说明：** 约 50,000 小时、论文报告 35.9M 选中片段的人类操作第一视角经验，供 HumanWAM 的纯人类及人机联合预训练研究。

## 数据来源与整理

- **ModelScope 官方镜像线索：** [EgoCorpus-50kH-LeRobot-v3](https://modelscope.cn/datasets/DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3) 出现在 HumanNet 的 Open-HumanWAM 分支 README；镜像数据卡和上传状态尚未独立读取，**不能认为全量可下载**。
- **HumanNet 子集计数：** Open-HumanWAM v2 表 1 报告参与此配方的 Ego 子集筛选前 **90,275.25 h**、后 **40,743.58 h**；五源累计筛选前 **104,289.41 h**，后 **50,000 h**。这是配方采用的特定子集，**不是 HumanNet 的 Ego 全库规模**。“20 万小时原始 Ego”链接及待核验事项见[独立归档](humannet-modelscope-ego.md)。


- 合并 HumanNet、EgoSuite-Open100K、EgoDex、Gen-HumanEgo、EgoVerse，涵盖家居、服务与工业操作。
- 对手部可见性、相机/动作稳定性和操作活跃度筛选，并将镜头坐标、腕轴与夹爪状态对齐为统一双臂 EEF 表示。
- 论文报道 **35.9M selected clips**；HF 数据卡称打包后为 **31,396,799 physical episodes**。二者计量单位不同，不能直接按差值判断“缺失片段”。

## 发布格式与复现边界

- **LeRobot v3**；含 MP4 压缩视频、Parquet 动作/状态、captions、episode 元数据和 provenance（来源追溯），不包含已缓存文本 embedding。
- 内容按 `egostandard/part-*`、`egopro/part-*`、`egodex/part-*`、`egoverse/part-*`、`humannet/part-*` 分组；每个 part 独立含 `data/`、`videos/`、`meta/`。humannet 组也保留 Gen-HumanEgo 等来源，通过 provenance 区分。
- 7 档已整理的 subset index：**100h / 500h / 1000h / 5000h / 10000h / 20500h / 50000h**。索引表列含 `dataset_path`、`episode_index`、`start_frame`、`end_frame`、`source`、`task`、`hand`；索引不是完整视频 payload。
- 官方说明小子集曾缺失的范围由 `egostandard/part-supp1105-*` 补齐，50K 选择集未变；**主数据上传仍在继续**。
- **访问限制：** 仓库地址公开，但 HF 要求登录并同意提供信息/使用条件后才能访问文件。多个原始来源各自的授权仍适用，不能把主代码仓库的 Apache-2.0 视为数据版权授权。

## 上传状态（有时效性）

| 官方数据卡快照 | 数字 |
|---|---|
| 更新时间 | **2026-11-10 11:27（UTC+8）** |
| 已上传且验证 | **23.24 TB / 26.67 TB（87.13%）** |
| 状态 | **上传中**（非完整发布） |
| 官方预测完成时间 | 2026-11-11 10:49（UTC+8）；仅为当时预测，不能当作承诺 |

此表只作**历史核对证据**。最新可用性、权限与体量以数据卡实时状态为准；截至本次核验，尚不能声称完整 50K payload 已可直接复现。

## 对 wiki 的映射

- [Open-HumanWAM 唯一论文/项目节点](../../wiki/entities/paper-open-humanwam.md)
- [HumanNet 原始数据项目](../../wiki/entities/humannet.md)
