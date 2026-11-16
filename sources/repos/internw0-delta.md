# InternW0-Δ — 官方代码仓库归档

- **仓库：** <https://github.com/InternRobotics/InternW0-Delta>
- **项目页：** <https://internrobotics.github.io/InternW0-Delta/>
- **默认分支：** main
- **核验日期：** 2026-11-16
- **许可：** 主代码 MIT；仓库 NOTICE 中的第三方依赖和模型/数据资产应各按其自身许可处理。
- **策展节点：** [InternW0-Δ](../../wiki/entities/paper-internw0-delta.md)

## 仓库当前开放范围

根目录已有 `src/wam`、`configs`、`deploy`、`eval`、`scripts`、`tools`、`tests`、`docs`、`LICENSE`、`NOTICE`、`README.md` 和中文 README。README 给出了安装、预训练、4D 蒸馏、目标任务 post-training、评测、真实机器人部署和可视化入口；不是只有 project-page 的演示仓。

安装说明要求 Linux、Python 3.11、CUDA GPU、FFmpeg；示例环境安装 PyTorch 2.10.0 / CUDA 12.8。预训练入口示例包括 `python tools/path_index.py`、`python tools/pretrain_text_cache.py`、`NPROC_PER_NODE=8 bash run.sh pretrain`；目标任务 post-training 有相应 task config。

## 当前发布状态需分开描述

- **训练/评测/部署实现：已公开**，MIT 代码许可。
- **预训练和任务模型权重：已发布。** Base、LIBERO、RoboTwin、RoboDojo 四个 Hugging Face 仓库均可见约 12.4 GB 对应权重文件，模型仓库标注 Apache-2.0，代码 MIT。官方 README 顶部的 coming soon 文案并非最新状态。
- **数据处理代码与数据：** 仓库开放相关处理入口，但原始和处理后数据能否下载/再分发取决于各自许可。
- **第三方模型：** Wan2.2、RynnBrain、Track4World 等组件不因本仓 MIT 声明而自动变为 MIT。

## 代码导航（README）

| 主题 | 官方入口 |
|---|---|
| 数据准备 | README 的 Data Preparation guide |
| 模型预训练与 4D 蒸馏 | README 的 Training、4D Distillation |
| 下游 post-training | LIBERO、RoboTwin、RoboDojo、EBench、RTC |
| 真机部署 | RTC 和同步/异步部署 guide |
| 推理加速 | LiteGen、VAE/VLM caching、MoT compile |
| 状态/动作接口 | `robot.yaml` 中统一 80D 带 mask 语义布局 |

## 源码调用链（2026-11-16 阅读核对）

1. **训练入口：** `run.sh` 转发 `scripts/train_distributed.sh`，按 task 组装 `accelerate launch` 执行 `scripts/train.py`；后者使用 Hydra 配置、`CacheManager.from_config`，根据缓存模式选择 `generate()` 或 `wam.runtime.run_training`。分布式脚本可用 `DRY_RUN=1` 查看组装命令。
2. **4D 蒸馏：** 官方指南先执行 `prepare_teacher.py`，然后 `cache.py select/extract/seal` 离线缓存 Track4World 特征，再运行 `bash run.sh 4D_distillation`；第三方教师许可证单独约束。
3. **部署：** `deploy/run.sh` → `deploy/run.py` → `deploy.policy.read_run/load_policy` → `deploy.robot.connect/ObservationValidator` → `deploy.controller.AsyncController`。每 16 步重新提交异步请求，控制循环 30 Hz。
4. **维度与安全边界：** `configs/pretrain/projections/robot.yaml` 定义 80D 带 mask 接口；当前官方双臂 ROS2 adapter 接受 14D 物理关节/夹爪命令，`deploy/controller.py` 核验 [32,80] 归一化计划及 [32,14] 执行动作。默认不带 `--execute` 时只推理不发布命令；部署时仍须底层限位和 watchdog。
5. **权重：** [Base](https://huggingface.co/InternRobotics/InternW0-Delta-Base/tree/main) · [LIBERO](https://huggingface.co/InternRobotics/InternW0-Delta-Libero/tree/main) · [RoboTwin](https://huggingface.co/InternRobotics/InternW0-Delta-RoboTwin/tree/main) · [RoboDojo](https://huggingface.co/InternRobotics/InternW0-Delta-RoboDojo/tree/main)。本次仅做源码核验，未实际训练或部署。
