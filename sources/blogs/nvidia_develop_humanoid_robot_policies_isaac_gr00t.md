# Develop Humanoid Robot Policies End-to-End with NVIDIA Isaac GR00T

> 来源归档（blog / NVIDIA Developer Blog）

- **标题：** Develop Humanoid Robot Policies End-to-End with NVIDIA Isaac GR00T
- **类型：** blog
- **作者：** Elizabeth Goodman（NVIDIA）
- **原始链接：** https://developer.nvidia.com/blog/develop-humanoid-robot-policies-end-to-end-with-nvidia-isaac-gr00t/
- **发布日期：** 2026-07-07
- **关联代码仓：** https://github.com/NVIDIA/Isaac-GR00T
- **入库日期：** 2026-07-13
- **一句话说明：** 介绍 **Isaac GR00T Development Platform** 如何把 Isaac Lab-Arena、Teleop、GR00T 1.7 后训练与 Isaac ROS 部署串成模块化端到端人形策略流水线，并以 G1 静态 pick-and-place 仿真教程示范五阶段 artifact 衔接。

## 核心摘录（归纳，非全文）

### 平台定位

- 解决人形开发管线 **工具孤岛、数据格式不兼容、手工胶水集成** 问题
- **开放且模块化**：可只用单组件，也可走完整 NVIDIA 栈
- GR00T 1.7 为 **首个 Apache 2.0 可商用的开源通才人形 VLA**（博客表述）；3B 基座 checkpoint 公开

### GR00T 1.7 博客侧亮点（相对 N1.6）

- 预训练：约 **32K 小时** 真人演示/egocentric + **8K 小时** 仿真（BEHAVIOR、RoboCasa、Simulated GR-1）
- **Cosmos-Reason2-2B** 替换 Eagle；原生宽高比图像编码
- **ONNX/TensorRT** 全链路导出；任务/子任务分解增强长视界推理
- 博客报告 benchmark 增益：DROID-F0 **+10%**、DROID-F6 **+61%**；SimplerEnv Bridge **+5%**、Fractal **+2%**

### 仿真 walkthrough（G1 静态 apple pick-and-place）

1. **Isaac Lab-Arena 搭环境** — 场景资产 + `PickAndPlaceTask` + embodiment + OpenXR 遥操作设备；WBC 选型影响训练分布（博客示例用 **AGILE WBC** 而非 stand/walk 控制器）
2. **Isaac Teleop 采 demo** — `record_demos.py` → HDF5（示例 400 条 trajectory，可分多 session 累积）
3. **HDF5 → LeRobot** — `convert_hdf5_to_lerobot.py` + YAML 字段映射（`state_name_sim` / `action_name_sim` / `pov_cam_name_sim`）
4. **GR00T 1.7 后训练** — 在 Isaac-GR00T 仓 `launch_finetune.py`；冻结 LLM、调 visual/projector/diffusion
5. **闭环评测** — Arena `policy_runner.py` + 远端 GR00T server（ZMQ）；可调 `--num_episodes` 估 success rate

### 生态采用（博客列举，非完整清单）

- 人形/AI 公司：1X、Agility、Skild AI 等集成 Isaac Teleop / Sim / Lab / ROS 组件
- 研究机构：Stanford、CMU、UCSD、ETH、AI2 等试验统一工作流
- XR 设备：PICO、HTC Vive、Manus 等支持 Isaac Teleop 采集

## 对 wiki 的映射

- [Isaac GR00T（开发平台）](../../wiki/entities/isaac-gr00t.md) — 端到端五阶段流程与 N1.7 GA 工程入口
- [GR00T N1（论文）](../../wiki/entities/paper-hrl-stack-34-gr00t_n1.md) — N1 论文机制与早期数据金字塔叙事
- [GR00T-WholeBodyControl](../../wiki/entities/gr00t-wholebodycontrol.md) — WBC 与 G1 全身控制细节
- [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md) — 仿真训练框架层
