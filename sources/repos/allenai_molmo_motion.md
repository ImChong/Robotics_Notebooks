# allenai/molmo-motion 仓库与 MolmoMotion-1M 数据卡精读归档

- **类型：** 开源仓库与数据集卡精读
- **仓库：** <https://github.com/allenai/molmo-motion>
- **数据集卡：** <https://huggingface.co/datasets/allenai/molmo-motion-1m>、`PointMotionBench`
- **已发布模型：** `allenai/MolmoMotion-4B-H3-F30`、`allenai/MolmoMotion-4B-H1-F32`
- **通读：** 已完成
- **核对范围：** `allenai/molmo-motion` 源码与文档（`README.md`、`examples/01_quickstart.py`、`src/molmo_motion/modeling.py`、`src/molmo_motion/processor.py`、`launch_scripts/sft.py`、`launch_scripts/eval_pointmotionbench.py`、`data_generation/` 目录结构）以及 HuggingFace `MolmoMotion-1M` 完整数据集说明（`LICENSE` 拆分与重构脚本契约）。

---

## 一、开源代码与已发布模型边界

1. **仅公开自回归（AR）变体**：`README.md` 明确指出本仓库代码及 HuggingFace 发布的 `MolmoMotion-4B-H3-F30`（`H=3, F=30`，面向 15 fps 2 秒视频预测）与 `MolmoMotion-4B-H1-F32`（单关键帧版本）均对应论文中的 **autoregressive（AR）** 版本，初始化基础训练需下载公开的 `Molmo2-4B-Pretrain` 权重；论文附录 D–E 中的 DiT Flow Matching 变体 `MolmoMotion-FM` 未随此仓库发布。
2. **三个注册 CLI 入口**：`pip install -e .[viz]` 安装后提供 `molmo-motion-train`、`molmo-motion-eval`、`molmo-motion-convert-hf`，官方标准配方直接通过 `torchrun --nproc-per-node=8 launch_scripts/sft.py` 与 `launch_scripts/eval_pointmotionbench.py` 调度。

---

## 二、核心源码链路核对

1. **推理接口（`MolmoMotionProcessor` → `MolmoMotion.predict_trajectory()`）**：
   - `processor.py` 中的 `_format_history_tracks()` 以末帧第 0 点作为公共 `anchor_3d`，将历史 3D 坐标相减并按 `_COORD_SCALE = 1000`（毫米）量化拼入 `<tracks coords="...">3d object history</tracks>` 提示词，同时整理多帧 RGB `images` 与可选 2D 网格点 `metadata[].coords_2d`。
   - `modeling.py` 的 `predict_trajectory()` 将预算设足至 `max_new_tokens = 160 * future_horizon`（避免 `<tracks>` 提前截断），调用底层 `Molmo2Trajectory.generate()`，再用 `parse_tracks_text()` 与 `tracks_to_array()` 解析坐标增量并加回 `anchor_3d`，返回米制 `(P, F, 3)` `future_3d`。单张 80 GB A100 加载耗时约 110 s、单次推理约 40 s；`examples/01_quickstart.py` 支持 `--from-prediction` 免 GPU 直接渲染内置预测的 2D 投影 MP4。
2. **两阶段训练与全点评测（`launch_scripts/sft.py` & `eval_pointmotionbench.py`）**：
   - 首次训练前须先运行 `python scripts/build_track_keys_cache.py` 校验本地重构轨迹；`sft.py` 使用 OmegaConf 组合 `TrainConfig`（FSDP2 + `amp_bf16` + 独立学习率与 `PackingConfig`）并调用 `run_trainer(conf)` 执行 Stage 1（40K 步 `trajectory_3d_human_p8_h3_f8`）与 Stage 2（10K 步长时程微调）。
   - `eval_pointmotionbench.py` 按 `hot3d,worldtrack,davis` 顺序驱动 `molmo_motion.eval.full_rollout.main()`（默认 `--all_points --max_points_per_clip 24` 分批滚动覆盖全物体表面点），最终 `_compute_metrics()` 统计 `ADE`、`FDE` 与阈值 `{0.01, 0.02, 0.05, 0.10, 0.20} m` 的平均 `PWT`，写入 `summary.json`。

---

## 三、MolmoMotion-1M 数据重构与许可契约

- HuggingFace `allenai/molmo-motion-1m` 不直接重分发受限原始视频（除 `molmospaces/` 含全部文件外），各子集分发标注 JSON（`annotations/` 为唯一真值源，含 `fps`、`num_frames`、`caption`、`clips_by_object`）、tar 分片轨迹/相机参数及 `reconstruct_*.py`；其中 `stereo4d/` 仅附带 `track_index/`，运行 cache 脚本前必须先本地重建 `tracks/` 与 `camera/`。
- 子集按上游许可分立：`egodex` / `ytvis` / `hdepic` / `xperience` / `stereo4d` 为 `CC BY-NC 4.0`，`droid` 与 `molmospaces` 为 `CC BY 4.0`；`data_generation/` 目录另开源了 `Molmo2` + `MolmoPoint` + `SAM 3` + `AllTracker` + `ViPE` 的自动标注流水线代码。
