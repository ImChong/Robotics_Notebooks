# AnyWorld（xpeng-robotics/AnyWorld）

> 来源归档（官方源码仓 · Apache-2.0）

- **标题：** AnyWorld
- **类型：** repo
- **机构：** 小鹏机器人（XPENG Robotics）等
- **代码仓：** <https://github.com/xpeng-robotics/AnyWorld>
- **项目页：** <https://xpeng-robotics.github.io/anyworld/>
- **论文：** <https://arxiv.org/abs/2608.29242>（[`sources/papers/anyworld_arxiv_2608_29242.md`](../papers/anyworld_arxiv_2608_29242.md)）
- **入库日期：** 2026-11-22
- **核对范围：** 核对 `xpeng-robotics/AnyWorld@2ed9d9c`（2026-09-02）仓库结构、`README.md`、`image_editing/infer.py`、`scripts/train_image_editor.sh`、`world_model/scripts/infer.py`、`scripts/infer_world_model.sh`、`world_model/fastvideo/entrypoints/video_generator.py`。未运行 GPU 推理。
- **通读：** 已完成
- **一句话说明：** AnyWorld 官方开源仓库（Apache-2.0），提供首帧具身编辑器（基于 DiffSynth-Studio / Qwen-Image-Edit-2511）训练与推理脚本，以及因子化动作–相机–具身世界模型推理管线（基于 FastVideo / Wan2.1 Combined-Control）。

## 开源状态与入口

| 模块 | 入口文件 | 依赖底座与作用 |
|------|----------|----------------|
| 首帧具身编辑训练 | `scripts/train_image_editor.sh` → `image_editing/training/train.py` | 基于 `DiffSynth-Studio` + `Qwen/Qwen-Image-Edit-2511`，用 `accelerate`（ZeRO-2）微调 `dit` 构造机器人首帧 |
| 首帧具身编辑推理 | `image_editing/infer.py` | 加载 `QwenImagePipeline` 与 `anyworld_image_editor.safetensors`，将人类首帧替换为 IRON 等机器人首帧并输出 `manifest_rank*.csv` |
| 世界模型多卡推理 | `scripts/infer_world_model.sh` → `world_model/scripts/infer.py` | 读取含 `caption`、`image_path`、`control_video_path`、`state_camera_extrinsic_path` 与相机内参的 `validation.json`，调用 `VideoGenerator.from_pretrained(..., workload_type="i2v_combined_control")` 生成机器人域视频 |

## 对 wiki 的映射

- [`wiki/entities/paper-anyworld.md`](../../wiki/entities/paper-anyworld.md)
- [`sources/papers/anyworld_arxiv_2608_29242.md`](../papers/anyworld_arxiv_2608_29242.md)
