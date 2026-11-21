# LingBot-Video

> 来源归档（国内具身开源全景）

- **标题：** LingBot-Video
- **类型：** repo
- **机构：** 蚂蚁灵波
- **链接：** https://github.com/Robbyant/lingbot-video
- **分类：** 世界模型
- **入库日期：** 2026-09-06
- **核对范围：** 2026-11-21 核对 `Robbyant/lingbot-video@dd5c231` 官方 `README.md`、`scripts/inference.py`、`lingbot_video/inference_backend.py`、`runner.py`、`pipeline_lingbot_video.py`、`scripts/multi-gpus/run_moe_t2v_refiner_fsdp_cp8.sh`，并阅读主论文 [arXiv:2607.07675v1](https://arxiv.org/html/2607.07675) HTML（摘要、§1 引言、§2 单流稀疏 MoE DiT/3D MM-RoPE/AdaLN-Single/级联 Refiner）；未运行 30B MoE 推理。
- **通读：** 已完成
- **一句话说明：** 蚂蚁灵波 开源项目 LingBot-Video（世界模型），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/cn-os-lingbot-video.md`](../../wiki/entities/cn-os-lingbot-video.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/cn-os-lingbot-video.md](../../wiki/entities/cn-os-lingbot-video.md)

## 官方资源补核（2026-10-05）

- **项目页：** <https://technology.robbyant.com/lingbot-video/>；代码 <https://github.com/robbyant/lingbot-video>。
- **机制：** Dense 1.3B / MoE 30B-A3B，70k+ 小时具身/网络视频；美学、物理与任务完成 reward 后训练。
- **发布边界：** 2026-09-18 发布 8-step DMD T2V / Ti2V checkpoint，不等于所有训练数据公开。
- **入口：** `scripts/inference.py`、`scripts/single-gpu/run_moe_dmd_t2v.sh` / `run_moe_dmd_ti2v.sh`、`rewriter/inference.py`；配置、权重与大显存要求需逐项匹配。
