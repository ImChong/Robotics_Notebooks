# VT-Collab/CASTER

> 来源归档（repo）

- **标题：** CASTER
- **类型：** repo
- **来源：** Virginia Tech Collab 实验室
- **代码：** <https://github.com/VT-Collab/CASTER>
- **论文：** <https://arxiv.org/abs/2610.12567>
- **项目页：** <https://collab.me.vt.edu/CASTER>
- **许可：** MIT
- **核查日期：** 2026-10-12（clone --depth 1，tip `1c25559`，提交日 2026-10-11；页面显示 8 次提交）
- **一句话说明：** CASTER 官方实现：合成视频生成与校验、特征轨迹提取、Isaac Sim 数字孪生 CEM 优化、真机回放。
- **沉淀到 wiki：** [`wiki/entities/paper-caster.md`](../../wiki/entities/paper-caster.md)

## 目录与入口（已 clone 核对）

- `src/caster/`：`video_gen.py`、`reconstruct.py`、`pipeline.py`、`feature_extract.py`、`visualize_dataset.py`；子目录 `llm/`（gemini、gpt_oss、video_validator）、`servers/`（sam3d_server、servers）、`sim_env/`（sim_server）、`optimization/`（run_optimization、replay、optimizer、spline、costs/feature|trajectory|obstacle、grasp、validation、rollout）、`real/`（capture_dataset、replay_trajectory、camera）、`config/`（Hydra 式 yaml，含每任务 video_gen_prompt）。
- 子模块：DA3、DEVA、TrackCraft3R、M2T2、IsaacLab、sam-3d-objects（独立环境）。
- 环境：`uv sync`；Isaac Sim 6.0.1 容器（`nvcr.io/nvidia/isaac-sim:6.0.1`）；`.env` 需 Gemini Robotics ER 与 OpenAI 兼容端点密钥。
- `cost_function=feature` 对应 CASTER，`cost_function=trajectory` 对应论文中 Mimic 基线。
- 本次未实际运行，仅核对结构与 README。
