# Genie-Envisioner-V1

> 来源归档（国内具身开源全景）

- **标题：** Genie-Envisioner-V1
- **类型：** repo
- **机构：** 智元机器人
- **链接：** https://github.com/AgibotTech/Genie-Envisioner-V1
- **分类：** 世界模型
- **入库日期：** 2026-09-06
- **一句话说明：** 智元机器人 开源项目 Genie-Envisioner-V1（世界模型），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/paper-sa-2508-05635-genie-envisioner-a-unified-world-foundation-plat.md`](../../wiki/entities/paper-sa-2508-05635-genie-envisioner-a-unified-world-foundation-plat.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/paper-sa-2508-05635-genie-envisioner-a-unified-world-foundation-plat.md](../../wiki/entities/paper-sa-2508-05635-genie-envisioner-a-unified-world-foundation-plat.md)

## 官方资源补核（2026-10-05）

- **项目页：** <https://genie-envisioner.github.io/>（已打开，本次未提取正文，运行边界以官方 README 核查）。
- **论文：** <https://arxiv.org/abs/2508.05635>
- **代码：** <https://github.com/AgibotTech/Genie-Envisioner-V1>，旧 `Genie-Envisioner` 地址重定向到此仓。
- **权重：** HF `agibot-world/Genie-Envisioner` 的 GE-Base；ModelScope `agibot_world/Genie-Envisioner` 内 GE-Act Calvin 与 GE-Sim Cosmos2 checkpoint。
- **入口：** `scripts/get_statistics.py`；`scripts/train.sh main.py` + `configs/ltx_model/{video_model_lerobot,policy_model_lerobot}.yaml`；`web_infer_scripts/run_server.sh`；`gesim_video_gen_examples/infer_gesim.py`。
- **许可：** 复用的 LTX/Cosmos/pipeline/openpi-client 目录 Apache-2.0；其余代码与数据 CC BY-NC-SA 4.0。
- **版本：** README 列 2026-05-28 GE-Sim 2、2026-09-10 GE-Act 2；这些是独立后续发布，不等于 V1 全面替换。
