# 英伟达开源宇树机器人 VLA 新版本：GR00T 1.7 用宇树 G1 跑通 VR 采集、LeRobot 后训练、仿真评测与部署流程

> 来源归档（blog / 微信公众号）

- **标题：** 英伟达开源宇树机器人 VLA 新版本：GR00T 1.7 用宇树 G1 跑通 VR 采集、LeRobot 后训练、仿真评测与部署流程
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/Y2mlKtd-dGGdA33Sx_sDCw
- **发表日期：** 2026-07-13
- **入库日期：** 2026-07-13
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；正文约 1.1 万字 / 5 图；`--no-images` 仅归档正文
- **原文出处：** 转载自 [NVIDIA Technical Blog — Develop Humanoid Robot Policies End-to-End with NVIDIA Isaac GR00T](https://developer.nvidia.com/blog/develop-humanoid-robot-policies-end-to-end-with-nvidia-isaac-gr00t/)（2026-07-07）；本仓库英文一手归档见 [nvidia_develop_humanoid_robot_policies_isaac_gr00t.md](./nvidia_develop_humanoid_robot_policies_isaac_gr00t.md)
- **项目链接：** https://github.com/NVIDIA/Isaac-GR00T
- **一句话说明：** 中文策展转载 NVIDIA GR00T 1.7 端到端平台介绍，以 **Unitree G1** 静态 apple pick-and-place 仿真链路为例，强调 **OpenXR VR + CloudXR 遥操 → HDF5 → LeRobot → GR00T N1.7 后训练 → Arena 闭环评测** 与 **AGILE WBC** 采数分布选择。

## 核心摘录（归纳，非全文）

### 平台五阶段（与官方 Table 1 对齐）

| 阶段 | NVIDIA 组件 | 功能 |
|------|-------------|------|
| 仿真环境 | Isaac Lab-Arena | 场景/任务/资产组合 |
| 数据采集 | Isaac Teleop（VR + CloudXR） | 高质量 demonstration |
| 策略训练 | Isaac GR00T 1.7 + 训练脚本 | 仿真+真机 demo 后训练 VLA |
| 策略评估 | Isaac Lab-Arena | 部署前仿真闭环 |
| 策略部署 | Isaac ROS + Jetson Thor | LEAPP 导出与端侧推理 |

### GR00T 1.7 文内强调点

- **预训练规模（博客口径）：** ~3.2 万小时真人演示/egocentric + ~8K 小时仿真（BEHAVIOR、RoboCasa、Simulated GR-1）
- **骨干：** Cosmos-Reason2-2B（Qwen3-VL），替换 N1.6 Eagle
- **许可：** Apache 2.0 可商用开源 VLA；基座 **3B**（`nvidia/GR00T-N1.7-3B`）
- **相对 N1.6 基准（文内数字）：** DROID-F0 **+10%**、DROID-F6 **+61%**；SimplerEnv Bridge **+5%**、Fractal **+2%**

### G1 静态 apple 仿真 walkthrough（策展工程要点）

1. **环境：** `galileo_g1_static_pick_and_place`；货架前双臂把 apple 移到 plate；**AGILE WBC**（静态任务，非 stand/walk 分解）
2. **WBC 选型警告：** 遥操时 AGILE WBC / PinkIK 生成的 **joint-space 目标即训练标签**；换控制器 = 换训练分布
3. **采数：** `record_demos.py` + OpenXR；示例 **400 条** trajectory（可多 session 累积；可先小批量试链路）
4. **转换：** `convert_hdf5_to_lerobot.py` + `g1_static_apple_config.yaml`（`robot_joint_pos` / `processed_actions` / `robot_head_cam_rgb`）
5. **后训练：** Isaac-GR00T 仓 `launch_finetune.py`；冻结 LLM，调 visual/projector/diffusion；`embodiment-tag new_embodiment`
6. **评测：** Arena `policy_runner.py` + 远端 GR00T ZMQ server；示例 metrics `success_rate: 1.0`（单 episode 冒烟）；规模化用 `--num_episodes 100/1000` + `--num_envs 5`

### 文内额外资源（相对英文 blog 归档）

| 资源 | 链接 |
|------|------|
| G1 仿真环境搭建完整代码（GitLab Pages） | https://unitree-g1-physical-ai-workflow-b42650.gitlab-master-pages.nvidia.com/simulation-workflow/sim-environment-code-review.html |
| NVIDIA Learning 端到端教程 | https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/index.html |
| GR00T 平台介绍视频 | https://www.youtube.com/watch?v=IwORUWekfxc |
| Isaac Teleop + GR00T 1.7 LeRobot 集成（HF Blog） | https://huggingface.co/blog/nvidia/nvidia-isaac-teleop-and-gr00t17-in-lerobot |
| GR00T 1.7 Brev Launchable | https://brev.nvidia.com/launchable/deploy?launchableID=env-3DgCELkan85JSOjYWxb3UlB0f5v |

### 生态列举（文内，非完整清单）

- 人形/AI：1X、Agility、Skild AI 等集成 Isaac Teleop/Sim/Lab/ROS
- 高校：Stanford、CMU、UCSD、ETH、AI2 试验统一 E2E 工作流
- XR：PICO、HTC Vive、Manus 等支持 Isaac Teleop 采集

## 对 wiki 的映射

- [Isaac GR00T（开发平台）](../../wiki/entities/isaac-gr00t.md) — 增补 G1 仿真教程链路与中文策展来源
- [GR00T N1（论文）](../../wiki/entities/paper-hrl-stack-34-gr00t_n1.md)
- [LeRobot](../../wiki/entities/lerobot.md) — LeRobot 格式与 HF `groot` 集成
- [GR00T-WholeBodyControl](../../wiki/entities/gr00t-wholebodycontrol.md) — AGILE WBC / SONIC 低层
