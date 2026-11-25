# GalaxyGeneralRobotics/RoboGesture（官方推理、部署与训练核心源码）

> 来源归档（ingest）

- **类型：** repo
- **仓库：** <https://github.com/GalaxyGeneralRobotics/RoboGesture>
- **项目页：** <https://robogesture.github.io/>
- **论文：** <https://arxiv.org/abs/2608.28693>
- **会议：** ECCV 2026
- **License：** Apache-2.0（部分三方资产/SDK 见 `licenses/` 与 `THIRD_PARTY_NOTICES.md`：`UNITREE_ASSETS_BSD-3-Clause.txt`、`UNITREE_SDK_BSD-3-Clause.txt`、`BRAINCO_REVO2_LICENSE.txt`）
- **版本核对：** `GalaxyGeneralRobotics/RoboGesture@2913f1d`（2026-11-21 克隆核对）
- **核对范围：** 已通读仓库 `README.md`、`robogesture/cli.py`、`robogesture/offline.py`、`robogesture/model.py`、`robogesture/motion.py`、`robogesture/collision.py`（文件头与配对表）、`robogesture/app.py`、`robogesture/session.py`、`robogesture/robot.py`（头部）、`training_examples/README.md` 与 `training_examples/demo.py`；说明文档未独立运行 GPU/真机或下载 LFS 权重
- **通读：** 已完成

## 公开范围与架构事实

1. **已公开组件：**
   - **推理与渲染 CLI（`robogesture/cli.py`）：**
     - `robogesture infer <wav> --output <out.npz>`（`offline.infer_wav`）：输入 WAV 经重采样转 24 kHz PCM → `MotionGenerator.generate`（2 块各 1 s 静音历史预热 + Mimi 2-codebook token + 50 步流匹配 DiT-CFM）生成 `[T, 41]` 上身与双手动作 → `model_to_robot` 映射到 G1 30 通道命令 → `smooth_model_blocks` → 默认经过 `CollisionFilter.filter`（固定底座零腿角场景）→ 输出 30 FPS 60 列 `qpos[T, 60]` NPZ 与对齐 `.wav`。
     - `robogesture render <npz> --output <out.mp4>`：基于 MuJoCo（EGL/OSMesa）输出 960×720 H.264/AAC 视频。
     - `robogesture stream --execute --confirm SEND_MOTION`（`app.run`）：完整英文整句对话流，连接 G1 DDS（FSM 501、`rt/lowstate` 与 BrainCo Revo2 双手）、`RobotAudioClient`、Volcengine ASR/TTS、`LocalAssistant`（`Qwen/Qwen3-4B-Instruct-2507` + `checkpoints/qwen_lora/`）与 `MotionSession`（整句碰撞检查 → 音频 staging → 30 Hz 同步下发）。
   - **Git LFS 推理权重与示例：**
     - `checkpoints/motion/`（54 类语义动作 checkpoint，41 维运动空间）
     - `checkpoints/qwen_lora/`（动作标签英文对话 LoRA adapter）
     - `examples/`：15 组配对音–动 `.wav` / `.npz` 示例
   - **训练核心示例（`training_examples/`）：**
     - `python -m training_examples.demo` 用合成张量演示 Stage 0（`10 * beat MSE + semantic CE`）与 Stage 1（手部权重 4、帧加权 flow loss + `0.5 * kinetic loss + 0.5 * semantic CE`，概率 `0.15` 整批历史置零、冻结 beat head）两阶段前向与梯度更新，不读写数据集或权重。
2. **未公开范围（README 与 `training_examples/README.md` 明确声明）：**
   - 论文全量 1000 h / 300+ 语义类数据集、原始 MoCap 重定向与标注预处理管线、完整训练/验证划分与大规模训练编排脚本、以及 Qwen/LoRA 训练脚本均未随本仓库发布。

## 对 wiki 的映射

- [paper-robogesture.md](../../wiki/entities/paper-robogesture.md)
- [robogesture_arxiv_2608_28693.md](../papers/robogesture_arxiv_2608_28693.md)
