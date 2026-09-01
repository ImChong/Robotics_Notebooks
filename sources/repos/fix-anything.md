# kvuong2711/fix-anything

> 来源归档

- **标题：** FixAnything（ECCV 2026 官方推理实现）
- **类型：** repo
- **组织 / 作者：** kvuong2711（Khiem Vuong, CMU）
- **代码：** <https://github.com/kvuong2711/fix-anything>
- **论文：** <https://arxiv.org/abs/2608.23549>
- **项目页：** <https://fix-anything.github.io/>
- **权重：** <https://huggingface.co/kvuong2711/fix-anything>
- **许可：** Apache-2.0（与 Wan2.1 一致）
- **入库日期：** 2026-09-01
- **一句话说明：** Wan2.1-I2V-14B + FixAnything LoRA 推理；支持退化渲染视频清理与 MapAnything 两视图重建→渲染→修复管线。

## 入口速查（对齐 README · 2026-09-01）

| 路径 / 命令 | 作用 |
|-------------|------|
| `pip install --no-build-isolation -e .` | 安装 FixAnything（基于 pinned DiffSynth-Studio） |
| `python scripts/download_models.py --model_dir checkpoints` | 下载 Wan2.1-I2V-14B + FixAnything LoRA |
| `python scripts/run_inference.py --input <video> --output_dir <dir>` | 对 61 帧退化渲染做清理；默认 `--clean_frame_indices "0 60"` |
| `python scripts/run_mapanything.py --images <folder> --output_dir <dir>` | MapAnything 重建 + 轨迹渲染 → `rendered.mp4` |
| `pip install --no-build-isolation -e ".[mapanything]"` | 可选：MapAnything + pyrender 自定义采集管线 |

## 项目页 / 源码开放核查（步骤 2.5）

- **状态：已开源（推理）** — 推理脚本、示例、HF 权重、Apache-2.0 许可。
- **边界：** SFT / Flow-DPO **训练代码未发布**；默认 50 步去噪，5 步可 10× 加速且质量接近（论文 Tab. 5）。

## 与本仓库知识的关系

- 论文归档：[`sources/papers/fixanything_arxiv_2608_23549.md`](../papers/fixanything_arxiv_2608_23549.md)
- 项目页：[`sources/sites/fix-anything-github-io.md`](../sites/fix-anything-github-io.md)
- wiki：[`wiki/entities/paper-fixanything.md`](../../wiki/entities/paper-fixanything.md)
- 基座：[`wiki/entities/paper-wan-video.md`](../../wiki/entities/paper-wan-video.md)
