# Hugging Face 模型卡：Cosmos3-Nano 四个交互式仿真 checkpoint

> 来源归档

- **类型：** site / huggingface-model-cards（4 个仓）
- **核查日期：** 2026-11-06（`https://huggingface.co/api/models/<id>` + `/raw/main/README.md` 通读；全部 `gated: false`）
- **关联博文：** [HF 博文归档](../blogs/nvidia_cosmos_interactive_world_simulation_hf_blog.md)
- **关联代码：** [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos)（入口 / 选型）、[NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework)（训练 / 推理；已浅克隆静态阅读，未运行）
- **对应 wiki：** [Cosmos Interactive World Simulation](../../wiki/entities/paper-cosmos-interactive-world-simulation.md)

## 共同事实（四卡一致）

- 许可：`license: other`，`license_name: openmdw1.1-license`，链接 <https://openmdw.ai/license/1-1/>；“ready for commercial or non-commercial use”。
- 版本：各卡均 **V1.0（2026-10-20）**；HF 与 GitHub 同日发布；仓内文件最近修改 2026-10-30（API `lastModified`）。
- 架构：Mixture-of-Transformers，源自 Cosmos3-Nano；卡面 “16B trainable parameters”；HF safetensors 元数据总参数 15.16B（Transfer-Auto / Sim-Auto 15,157,324,096；Sim-Transfer 15,156,222,144；Sim-Bimanual 15,173,136,576）——卡面 16B 与元数据约 15B 口径不同，不矛盾地并列记录。
- 运行时：PyTorch、vLLM-Omni、SGLang；硬件 Blackwell 与 Hopper；操作系统仅 Linux 经测试；示例均带 `--no-guardrails`（关闭 Cosmos3 文本 / 视频安全检查），卡面责任条款要求部署方自行加护栏。
- 后训练数据概览（Sim-Transfer、Sim-Bimanual 卡文字一致）：1.3B data points、393 个数据集条目、2024–2026 采集；私有数据集表中 AgiBot 仅 0.2M 样本。这是 **Cosmos 3 Omni 级别共享的数据说明**，不能当成 Bimanual 专属数据量（早先技术报告归档把它记在 Bimanual 名下，已更正）。

## 逐卡要点

| 项 | Transfer-Auto | Sim-Auto | Sim-Transfer | Sim-Bimanual |
|----|---------------|----------|--------------|--------------|
| 仓库 | nvidia/Cosmos3-Nano-Transfer-Auto | nvidia/Cosmos3-Nano-Sim-Auto | nvidia/Cosmos3-Nano-Sim-Transfer | nvidia/Cosmos3-Nano-Sim-Bimanual |
| 定位 | 双向、离线合成多传感器数据 | Transfer-Auto 的因果蒸馏流式版 | 因果蒸馏，仿真器空间控制 → RGB | 因果蒸馏，机器人动作 / 相机轨迹 → 视频 |
| 输入 | 文本 + 每相机 WSM 视频；可选首帧 RGB；LiDAR 需对齐 HD-map range-map，可选 1 帧实测 LiDAR | 同左；实测 LiDAR 条件仅支持 1 个初始扫描 | 控制视频（深度 / 分割 / 边缘 / 模糊）+ 文本 | 初始图像 + 动作 `(T, D)` / 相机轨迹 + 可选文本 |
| 输出 | 最多 11 路同步视频 + LiDAR 点云；至多 300 帧 / 视图，480p 或 720p，10–30 fps | 同左，分块流式 | 至多 600 帧，480p，30 fps | 480p，30 fps |
| 本体 / 装备 | Hyperion 8.1 相机组 1–11 路 + 128 线 LiDAR；其他装备需后训练 | 同左；自定义装备先后训练 Transfer-Auto 再蒸馏 | 面向 Isaac Lab 等仿真循环；最多 8 路并行环境（单 GB200，作者表述） | **卡面 Inputs 写 "Supported embodiment: AgiBot"**；博文、overview 视频说明、HF 示例文件（`assets/sim_bimanual/yam/`）、仓库代码（`abc_yam` / `molmoact2_yam` / `xdof_yam` 域）均含 YAM |
| 采样 | 35 步去噪 + CFG（示例 guidance 3，flow_shift 10） | 固定 4 步，guidance 1 | 蒸馏少步；示例 30 帧历史窗口 + 3 个 attention sink | 固定 4 步；64 潜帧滚动 KV + 8 sink |
| 吞吐（vLLM-Omni 单 GPU，卡面） | 201 帧、35 步：单视图 480p GB200×1 端到端 1m22s；7 视图 480p 11m20s；11 视图 720p 77m31s（无 LiDAR） | 无 LiDAR，GB200：1 视图 480p 16.8 fps，7 视图 480p 3 fps，11 视图 720p 0.74 fps；有 LiDAR 7 视图 480p 2.2 fps（单位口径卡面未细分是否按视图） | 480p：GB200 24.5、H100 14.6、H200 14.9、H20 4、RTX Pro 6000 8.3 fps | 480p：GB200 54.1、H100 32.6、H200 32.4、H20 10.7、RTX Pro 6000 20.6 fps |
| 许可数据说明 | 视频 1 万–100 万小时；LiDAR <1 万小时 | 同 Transfer-Auto | 视频 <1 万小时 | 视频 + 动作 / 相机轨迹 1 万–100 万小时 |

## 基准（卡面，作者自测）

- **Transfer-Auto（720p，guidance 3；基线 Cosmos-Transfer2.5-2B Multiview 用不同提示词）**：T2V Frame FID 45.5 对 61.9、FVD(I3D) 488 对 772、DOVER 9.01 对 8.56；I2V FID 27.8 对 41.9、FVD(I3D) 238 对 383、DOVER 8.27 对 8.22；I2V 的 CSE2 2.25 对 2.45、CSE5 1.80 对 1.98。LiDAR（T2V）：BEV JSD global 0.0401、Frechet Utonia 12.8、Object detection F1@3m 0.237，对比 LidarVerse（116 scans）0.116 / 44.8 / 0.200。
- **Sim-Auto（720p，guidance 1，7 相机 297 帧，4 步）**：T2V FID 48.6、FVD(I3D) 480、DOVER 7.30；I2V FID 33.6、FVD(I3D) 325、DOVER 7.51；跨视图 Sampson 误差 T2V 为 CSE2 2.60 / CSE5 2.24，**高于**基线 2.51 / 2.07。LiDAR：BEV JSD global 0.0552、Frechet Utonia 20.1、F1 0.188（低于 LidarVerse 的 0.200）。
- **Sim-Transfer（PAI-Bench-C，对比 Cosmos3-Nano）**：分割 mIoU 0.738 对 0.643；模糊 SSIM 0.887 对 0.856；边缘 F1 0.341 对 0.214；深度 SI-RMSE 0.619 对 0.842（越低越好）；DOVER 10.18 对 10.36、Prompt Following 7.03 对 7.18、Temporal Sampson error 0.0965 对 0.0810（均略逊于基座）。
- **Sim-Bimanual**：PAI-Bench-G I2V 相机导航——视频质量 73.91（LingBot-World 2.0 为 65.36、Sana-WM 65.21）、ATE 4.707（LingBot-World 2.0 为 4.170）、RRE 0.1002；Sana-WM-Bench simple 视频质量 77.29（Sana-WM 66.06）、hard 76.40（Sana-WM 65.57）；AgiBot-World——视频质量 76.09 / PSNR 14.41 / SSIM 0.684 / LPIPS 0.345，对比 DreamDojo 14B 的 74.94 / 12.36 / 0.664 / 0.359；AgiBot-GEAR——74.49 / 14.02 / 0.617 / 0.406，对比 DreamDojo 14B 的 73.00 / 12.11 / 0.582 / 0.433。

## 快速开始（卡面，只读摘录，未运行）

- 都是 `docker run ... vllm/vllm-omni:cosmos3` 起服务，再用 Python 客户端 POST `/v1/videos/sync`（SGLang 版用 `/v1/videos` 提交异步任务并轮询）。
- 管线类名：`Cosmos3MultiviewPipeline`（Transfer-Auto）、`Cosmos3NanoSimAutoPipeline`、`Cosmos3NanoSimTransferPipeline`、`Cosmos3NanoSimBimanualPipeline`；Sim-Auto / Bimanual 另需 `--deploy-config`（滚动 KV）。vLLM-Omni 侧源码不在本次读过的两个仓库内，未核对。
- Sim-Auto 快速开始仍使用预渲染的 WSM 控制视频做自回归 rollout，并非真正的闭环；闭环需要自备仿真运行时 + WSM 渲染器（博文与卡面均如此说明）。
