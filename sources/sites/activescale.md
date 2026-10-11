# ActiveScale 官方项目页与发布资源

- **类型：** site / model / dataset
- **标题：** ActiveScale: Scaling Active Perception for Robots across Model, Data, and Hardware
- **项目页：** <https://active-scale.github.io/>
- **论文：** <https://arxiv.org/abs/2609.18514>；[归档](../papers/activescale_arxiv_2609_18514.md)
- **代码：** <https://github.com/ShuaiZhou302/ActiveScale>；[归档](../repos/activescale.md)
- **模型：** <https://huggingface.co/davidzhou302/ActiveScale>
- **数据集：** <https://modelscope.cn/datasets/shuai302/ActiveScale>
- **初次入库：** 2026-09-17；**重新核查：** 2026-10-10
- **开放结论：** 代码、模型已发布，任务数据目录可获取；未核验整套 1000 小时语料完整发布。

## 一手页面核查

项目页链接 GitHub/ModelScope/Hugging Face；五任务 demo 标 autonomous rollout，移动展示另标 single-operator teleoperation。

平均 SR/TP：基座直接任务适配 30.0%/41.6%，完整方案 70.0%/78.4%；同架构无中训 62.0%/67.7%，history-only SR 37.0%。每类 150 示范/20 rollout；总增益不只是相机改动。

当前网页列人类 EgoLive/EgoVerse/EgoSuite；v2 只列前两者，分材料记录，不推断第三者的 v2 占比。

## 实际目录核查

- Hugging Face 公共模型 API：快照 `e9e8f8918a3cfd63580ed07625bb6e0739674a33`、gated=false；六目录 midtraining/bag/drawer/pot/box/under 有 safetensors。五任务有 norm stats，未见 midtraining assets；模型卡标 Gemma 条款。未下载大权重。
- ModelScope 公共 repo tree API：backpack/box/drawer/pot/under_table，目录 revision `d33e67c8aadd50027b375fbaf62eb161c1b59e75`。只核查目录，不宣称核验样本数、全部下载与许可。
- 2026-09-17 原归档待发布；本次更正当前状态，保留历史日期，不改原始公众号抓取。

## 对 wiki 的映射

- [ActiveScale 唯一项目节点](../../wiki/entities/paper-activescale.md)。
- [9 篇技术地图](../../wiki/overview/perception-action-transfer-9-papers-technology-map.md) — 当前开放状态。
