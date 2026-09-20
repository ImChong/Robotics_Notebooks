# HandEdit

> 来源归档

- **标题：** HandEdit
- **类型：** repo
- **链接：** <https://github.com/HandEdit/HandEdit>
- **项目页：** <https://handedit.github.io/>
- **论文：** <https://arxiv.org/abs/2608.12122>
- **数据集：** <https://huggingface.co/datasets/HandEdit/HandEdit>
- **机构：** 复旦大学（Fudan）· 上海交通大学（SJTU）· 香港大学（HKU）等
- **入库日期：** 2026-09-20
- **一句话说明：** HandEdit 基准的官方 **评测工具链** 与 Harmonizer 推理包装；`build_manifest.py` + `eval.py` 输出 PSNR/SSIM/LPIPS/FID 与 embodiment-aware 指标。

## 仓库入口（README，2026-09-20）

| 组件 | 说明 |
|------|------|
| 环境 | `conda create -n handedit-eval python=3.10` + `pip install -r requirements.txt` |
| 外部权重 | DINOv2、CLIP 需自行下载并传入 `--shape-model` / `--clip-model` |
| Manifest | JSONL 每行：`src_path`、`pred_path`、`gt_path`、mask 路径、`replacement_scope`、`target_name`、`urdf_ref_paths` 等 |
| 构建 manifest | `python build_manifest.py --src-root ... --out-manifest manifests/...jsonl` |
| 评测 | `python eval.py --manifest ... --experiment ... --output-dir runs` |
| Harmonizer | `harmonizer/` 目录：轻量后处理 checkpoint 与推理包装 |
| 数据集 | 完整 200M+ 实例见 Hugging Face，不在 git 仓内 |

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-handedit](../../wiki/entities/paper-handedit.md) | 论文实体、双轨协议与 baseline 读法 |
| [handedit-github-io](../sites/handedit-github-io.md) | 项目页与伪 GT 质量审计 |
| [paper-notebook-egodex-learning-dexterous-manipulation-from-larg](../../wiki/entities/paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md) | 五源之一 EgoDex |
