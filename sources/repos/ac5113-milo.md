# MILO（ac5113/MILO）

- **URL：** <https://github.com/ac5113/MILO>
- **组织：** ac5113（Agniv Chatterjee）
- **关联项目页：** [MILO 项目页](../sites/ac5113-milo.md)
- **关联论文：** [MILO 论文摘录](../papers/milo_arxiv_2608_27407.md)
- **入库日期：** 2026-08-31
- **再核日期：** 2026-09-05

## 一句话说明

MILO 官方实现：单图人—物交互三维重建；LRM 网格解释 + SMPL-H / 可选物体模板拟合（ECCV 2026）。

## 仓库快照（2026-09-05）

| 项 | 内容 |
|----|------|
| 默认分支 | `main`（创建 2026-06-20；代码推送至 2026-08-28） |
| 入口 | `milo/pipeline/run_pipeline.py`；步骤在 `milo/pipeline/steps/` |
| 文档 | `docs/PIPELINE.md`、`docs/INSTALL.md`、`docs/DATA.md`、`milo/pipeline/README.md` |
| Demo | `demo/example`（人推行李车）；输出 `fitted_human.obj` / `segmented_object.obj` |
| 评测 | `milo/eval/prepare_dataset.py` + `scripts/eval_results.sh`（PA-CD：`icp` / `template`） |
| 许可证 | **MIT**（仓内 `LICENSE`）。第三方 submodule 与人体模型保留各自许可 |
| 依赖 | `scripts/install_milo.sh`：PyTorch 2.6 + CUDA 12.6、Hunyuan3D-2、SAM 3、SAM 3D Objects、Grounded-SAM-2、HaMeR / ViTPose / HMR2.0、vendored Fast-Robust-ICP |

## 运行时入口（对齐 wiki 时序图）

核心步骤：`auto_masks` → `run_lrm` → `render` → `img_segment` / `mesh_segment` → `kp2d` / `triangulate` → `init_smpl` / `fit` → `isolate` → `collate` / `render_final`。

可选 `--template` 追加：`tmpl_render` → `correspond`（独立 `geo-aware` 环境，GeoAware-SC）→ `template_align`。

LRM 开关：`--lrm hy3d`（默认 Hunyuan3D-2.0）或 `--lrm sam3d`（SAM 3D Objects，HF 门控，官方写 ≥32 GB VRAM）。分割：`--segmenter sam3`（默认）或 `gsam2`。

最短命令：

```bash
python milo/pipeline/run_pipeline.py --data_root demo --seq example \
    --object "trolley" --object_prompt "a green trolleycase"
```

## 开源边界

- **已开源、可跑：** 训练/推理意义上的完整解释管线 + demo + 三套基准准备脚本。
- **不随仓：** SMPL-H / SMPL-X / VPoser / MANO 需官网注册；SAM 3 与 SAM 3D Objects checkpoint 需 Hugging Face 申请。
- **不算机器人策略：** 输出是网格，不是关节指令。

## 交叉链接

- [MILO 论文实体](../../wiki/entities/paper-milo.md)
