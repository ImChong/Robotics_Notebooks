# GraspVLA

> 来源归档（国内具身开源全景）

- **标题：** GraspVLA
- **类型：** repo
- **机构：** 银河通用
- **链接：** https://github.com/PKU-EPIC/GraspVLA
- **分类：** VLA/操作模型
- **入库日期：** 2026-09-06
- **一句话说明：** 银河通用 开源项目 GraspVLA（VLA/操作模型），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/cn-os-graspvla.md`](../../wiki/entities/cn-os-graspvla.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/cn-os-graspvla.md](../../wiki/entities/cn-os-graspvla.md)

## 官方资源补核（2026-10-05）

- **代码：** <https://github.com/PKU-EPIC/GraspVLA>；项目资源以此规范仓 README 为准。
- **权重：** <https://huggingface.co/vegebirrd/GraspVLA>。
- **数据：** SynGrasp-1B，README 2026-08-19 公告已发布；不能沿用早期“待发布”说明。
- **入口：** `uv sync --locked`；`vla_network.scripts.serve`、`vla_network.scripts.offline_test`；checkpoint 同目录需有 `config.json`、`preprocessor.npz`；真实控制接口须另适配。
- **机制：** 合成抓取动作与互联网语义共同学习，感知思维链连接语义理解与 flow 动作头。
