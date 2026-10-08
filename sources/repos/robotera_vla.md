# robotera_vla

> 来源归档（国内具身开源全景）

- **标题：** robotera_vla
- **类型：** repo
- **机构：** 星动纪元
- **链接：** https://github.com/roboterax/robotera_vla
- **分类：** VLA/操作模型
- **入库日期：** 2026-09-06
- **一句话说明：** 星动纪元 开源项目 robotera_vla（VLA/操作模型），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/cn-os-robotera-vla.md`](../../wiki/entities/cn-os-robotera-vla.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/cn-os-robotera-vla.md](../../wiki/entities/cn-os-robotera-vla.md)

## 官方 README 补核（2026-10-08）

- **总入口：** https://github.com/roboterax/robotera_vla/blob/main/README.md
- **训练说明：** https://github.com/roboterax/robotera_vla/blob/main/training/README.md

当前 `release_1.0` 以 M7 为默认本体，公开 `data_collection/`、`training/`、`inference/` 三部分。根 README 明确机器人侧已有的遥操作和 recorder 服务不在本仓实现。

训练基线来自 Physical Intelligence 的 π₀.₅ / openpi；训练 README 链接 `roboterax/M7_pickplace_example_ckpt` 与 `roboterax/M7_pickplace_example`，并提供归一化与微调命令，但 Status 仍写训练细节待负责人补充。MIT 许可按根 README / LICENSE 读取；代码、样例权重/数据链接与完整产品能力分别判断。这里不是 ERA-42 的完整预训练、权重和真机闭环开放记录。未确认仓库单一首发日期；README 修改日不作为模型发布日。
