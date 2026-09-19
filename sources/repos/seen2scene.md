# Seen2Scene（quan-meng/seen2scene）

- **URL：** <https://github.com/quan-meng/seen2scene>
- **项目页：** <https://quan-meng.github.io/projects/seen2scene/>
- **关联论文：** [seen2scene_arxiv_2603_28548.md](../papers/seen2scene_arxiv_2603_28548.md)
- **许可证：** MIT

## 一句话说明

ECCV 2026 **visibility-guided flow matching** 3D 场景补全/生成官方实现：稀疏 TSDF VAE + sparse transformer generator + ControlNet completion；依赖 fVDB、TorchSparse、FlashAttention。

## 运行时入口（README 对齐）

| 阶段 | 命令 |
|------|------|
| 环境 | `conda env create --name seen2scene --file env/sg.yml` + FlashAttention / fVDB / TorchSparse 安装脚本 |
| 样本数据 | `hf download MQ66/seen2scene-FRONT-3D --repo-type dataset --local-dir .` |
| 权重 | `hf download MQ66/seen2scene --local-dir .` |
| 训练 VAE | `python -m seen2scene.main vae ...` |
| 训练 generator | `python -m seen2scene.main generator --ae-log AE_LOG ...` |
| 训练 ControlNet | `python -m seen2scene.main control --ae-log AE_LOG --gen-log GEN_LOG ...` |
| Layout 生成 | `python -m seen2scene.main generator task:generation ...` |
| 扫描补全 | `python -m seen2scene.main control task:completion ...` |

**Released checkpoint 路径：** `AE_LOG=2025-12-19_01-23-28-525`，`GEN_LOG=2026-02-23_16-22-25-152`，`CONTROL_LOG=2026-02-26_14-01-23-930`。

## 交叉链接

- [paper-seen2scene 论文实体](../../wiki/entities/paper-seen2scene.md)
- [项目页归档](../sites/seen2scene-project.md)
- VDBFusion fork：<https://github.com/quan-meng/vdbfusion>
