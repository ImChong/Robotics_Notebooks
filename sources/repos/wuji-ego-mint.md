# wuji-ego-mint

> 来源归档（ingest · MINT 官方仓库）

- **标题：** wuji-ego-mint
- **类型：** repo
- **机构：** 舞肌科技（Wuji Technology）；上海科技大学；清华大学；香港大学；浙江大学
- **链接：** <https://github.com/wuji-technology/wuji-ego-mint>
- **模型：** <https://huggingface.co/ZZJAsher/mint_v1>
- **数据集：** <https://huggingface.co/datasets/ZZJAsher/wuji_ego_mint>
- **论文：** <https://arxiv.org/abs/2609.04958v1>
- **项目页：** <https://1847540790.github.io/mint-project-page/>
- **许可：** MIT
- **入库日期：** 2026-09-07
- **一句话说明：** MINT 官方实现：统一 egocentric 相机+双手世界系重建模型、EgoPipeline 伪标签管线、Web Viewer 推理、训练与 benchmark CLI，以及 Wuji Hand MuJoCo retargeting 参考。

## 开源状态

- **已开源**：推理 Web Viewer、训练代码、评测、EgoPipeline 编排与 **1,021 h** 结构化数据集（非视频部分）均已发布。
- **硬件：** 推理需 NVIDIA GPU **≥ 24 GB** VRAM。
- **第三方：** MANO 需用户在 [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) 自行下载；部分 HaWoR 适配权重受许可限制不可再分发（`THIRD_PARTY_NOTICES.md`）。

## 仓库布局（主入口）

| 路径 | 作用 |
|------|------|
| `mint/viewer` | Web Viewer：加载 checkpoint、MP4/LeRobot episode 推理与可视化 |
| `mint/inference` | 推理核心 |
| `model_train/` | 两阶段训练 |
| `eval/model_effect/benchmark/` | HOT3D / ARCTIC 等 benchmark CLI |
| `ego_pipeline/` | EgoPipeline 编排、清洗、LeRobot 导出参考 |
| `eval/simulate/wuji-retargeting/` | Wuji Hand URDF/MJCF 与 retargeting |
| `scripts/create_env.sh` | 创建 `mint-inference` conda 环境 |
| `scripts/download_assets.sh` | 下载公开权重与样例 |

## 快速复现路径（README 摘要）

```bash
git clone https://github.com/wuji-technology/wuji-ego-mint.git
cd wuji-ego-mint
bash scripts/create_env.sh inference
conda activate mint-inference
bash scripts/download_assets.sh
# 将 MANO_RIGHT.pkl / MANO_LEFT.pkl 放入 assets/mano/
# 启动 Web Viewer（见 README）
```

## 对 wiki 的映射

- [wiki/entities/paper-mint-ego-world-space-camera-hand-motion.md](../../wiki/entities/paper-mint-ego-world-space-camera-hand-motion.md)
- [wiki/entities/wuji-robotics.md](../../wiki/entities/wuji-robotics.md)
