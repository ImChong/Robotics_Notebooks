# facebookresearch/metamotivo（Meta Motivo 官方仓库）

> 来源归档（ingest · arXiv:2504.11054 配套官方实现）

- **标题：** Meta Motivo Official Implementation
- **类型：** repo
- **链接：** <https://github.com/facebookresearch/metamotivo>
- **项目页：** <https://metamotivo.metademolab.com/>
- **配套论文：** [Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models（arXiv:2504.11054）](../papers/bfm_awesome_metamotivo_arxiv_2504_11054.md)
- **环境仓库：** <https://github.com/facebookresearch/humenv>（`HumEnv`）
- **许可证：** CC BY-NC 4.0（非商用研究许可）
- **入库日期：** 2026-11-21
- **通读：** 已完成
- **核对范围：** 2026-11-21 通读 `README.md`、27 文件完整目录树、`examples/fbcpr_train_humenv.py`、`metamotivo/fb_cpr/agent.py` 与 `metamotivo/fb_cpr/huggingface.py`。
- **一句话说明：** Meta FAIR 发布的 FB-CPR 官方 PyTorch 实现，含 `HumEnv` 训练入口、6 个 Hugging Face 预训练权重（5 个 24.5M `metamotivo-S-*` 与 1 个 288M `metamotivo-M-1`）及零样本评估封装。

## 仓库入口与核心模块（2026-11-21 核查）

| 路径 | 职责 |
|------|------|
| `examples/fbcpr_train_humenv.py` | 主训练脚本：`Workspace` 构建 `make_humenv`（50 并行环境，`MoCapAndFall` 初始化）、`TrajectoryBuffer` + `DictBuffer`，循环运行 `FBcprAgent.update()` 与 `RewardEvaluation` / `TrackingEvaluation` |
| `metamotivo/fb_cpr/agent.py` | `FBcprAgent`（继承 `FBAgent`）：`encode_expert()` 用 `B` 编码专家片段、`update_discriminator()`（WGAN-GP）、`sample_mixed_z()`（60% 轨迹 / 20% online 目标 / 20% 均匀超球面）、`update_fb()`、`update_critic()`、`update_actor()` |
| `metamotivo/fb_cpr/huggingface.py` | `FBcprModel.from_pretrained("facebook/metamotivo-S-1")` 加载 24.5M/288M 权重与零样本推理接口（`reward_wr_inference` / `goal_inference` / `tracking_inference` / `act`） |
| `metamotivo/wrappers/humenvbench.py` | `RewardWrapper` 与 `TrackingWrapper` 对接 `humenv.bench` |

## 对 wiki 的映射

- [`wiki/entities/paper-bfm-02-metamotivo.md`](../../wiki/entities/paper-bfm-02-metamotivo.md)
