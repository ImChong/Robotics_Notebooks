# MAMMA

> 来源归档

- **标题：** MAMMA（Markerless Accurate Multi-person Motion Acquisition）
- **类型：** repo
- **来源：** MPI-IS Perceiving Systems（cuevhv）
- **链接：** <https://github.com/cuevhv/mamma>
- **Stars：** ~428（2026-06）
- **入库日期：** 2026-06-09
- **一句话说明：** CVPR 2026 Oral 官方实现：多视角 markerless mocap 管线（分割 → MammaNet 稠密 landmark → 跨视角匹配 → SMPL-X 优化），含 CLI、浏览器 GUI 与 iPhone 四相机 demo。
- **沉淀到 wiki：** [`wiki/entities/paper-mamma-markerless-motion-capture.md`](../../wiki/entities/paper-mamma-markerless-motion-capture.md)

---

## 核心定位

开源 **学术级 markerless 多人体 SMPL-X 采集** 复现入口：面向 **双人近距离交互**（舞蹈、拥抱、搬运等），强调 **全自动**（相对 Vicon 贴标与手工清 marker），输出可直接用于动画、生物力学或下游 **motion retargeting** 的 SMPL-X 时序。

## 管线步骤

| 步骤 | 模块 | 说明 |
|------|------|------|
| `ma_cap` | capture | 加载多视角序列与标定 |
| `ma_masks` | segmentation | SAM + YOLO 每人 mask，跨帧跟踪 |
| `ma_2d` | landmarks | MammaNet：512 稠密 2D landmark + σ + 可见性 + 接触 |
| `ma_3d` | optimization | 多阶段 L-BFGS 拟合 SMPL-X |
| `ma_vis` | visualization | 相机叠加 + Rerun 交互场景 |

入口：`python -m inference run`（`inference/cli/run.py`）。

## 依赖栈

| 组件 | 说明 |
|------|------|
| 环境 | micromamba/conda，CUDA；见 `docs/INSTALL.md` |
| 分割 | SAM 2 / SAM 3、YOLOv12、Detectron2 |
| 人体模型 | SMPL-X（需单独注册下载） |
| 骨干 | ViTPose-B 初始化；训练用 BEDLAM + MAMMASyn |
| 可视化 | Rerun |
| 训练 | PyTorch Lightning、Hydra、WebDataset |

## 仓库要点

- **许可：** 非商业科研（见 LICENSE）；SMPL-X 等第三方数据遵循各自许可
- **Demo：** `bash data/download_example.sh` + `configs/examples/presets/quick.yaml`（~5 min smoke）
- **自有数据：** 需标定 YAML + 序列目录布局，见 `docs/YOUR-DATA.md`
- **GUI：** `gui/scripts/dev.sh` 与 CLI 共用同一 runner
- **数据集下载：** 项目页账号 + `data/download_*.sh` 或 GUI 面板
- **待发布：** 评测脚本与处理后 eval 集（README TODO）

## 对 wiki 的映射

- 论文归档：[sources/papers/mamma_arxiv_2506_13040.md](../papers/mamma_arxiv_2506_13040.md)
- 项目页：[sources/sites/mamma-tue-mpg-de.md](../sites/mamma-tue-mpg-de.md)
- 实体页：[wiki/entities/paper-mamma-markerless-motion-capture.md](../../wiki/entities/paper-mamma-markerless-motion-capture.md)
