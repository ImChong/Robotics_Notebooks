# GALATEA 官方代码仓库（boyuan-an/GALATEA）

- **URL：** <https://github.com/boyuan-an/GALATEA>
- **论文：** [arXiv:2609.10050](https://arxiv.org/abs/2609.10050)
- **项目页：** <https://boyuan-an.github.io/GALATEA/>
- **开源：** 已开源（2026-11-03 `commit 36b9048` 已发布 `recon/`、`rl/` 源码及 `rl-v1.0` 预训练权重/示例数据；注意 `recon/` 为 NVIDIA 非商业许可、`rl/` 为 GPL-3.0）
- **核对范围：** 通读 `boyuan-an/GALATEA@36b9048` 根目录 `README.md`、`recon/README.md`、`rl/README.md`、`recon/scripts/run_example.sh`、`recon/scripts/run_pipeline_custom.sh`、`rl/scripts/train_4x6144.sh` 与 `rl/scripts/eval.sh`；未下载第三方 MANO/模型资产或在本地 GPU 运行。
- **通读：** 已完成

## 仓库入口与结构（commit `36b9048`）

| 组件 | 入口与依赖 | 许可与边界 |
|------|------------|------------|
| `recon/`（HOI 重建） | `scripts/run_example.sh`（自带 169 帧电钻示例）与 `scripts/run_pipeline_custom.sh`（MoGe-2 → GroundingDINO/SAM 2 → FoundationPose → WiLoR → HOI-DETR → 联合优化导出 `_phase3/.../gr_hsz_world.npy` 与 `gr_hsz_handverts_world.npy`） | NVIDIA License（限非商业研究）；需另行获取 MANO 及各感知模型权重 |
| `rl/`（Isaac Gym RL 跟踪） | `preprocess/run.sh`（质量清洗→重采样→workspace 放置→FR3 IK→8×减速→安全检查→CoACD 凸分解）→ `scripts/train_4x6144.sh` / `train_2x12288.sh` → `scripts/eval.sh` | GPL-3.0（基于 ManipTrans / SimToolReal / SAPG / IsaacGymEnvs）；`rl-v1.0` Release 提供 `power_drill`、`dog`、`old_camera2` 权重与数据 |
