---
type: entity
title: FEAP Vision MuJoCo 部署（深度 + TorchScript）
tags: [humanoid, mujoco, depth, torchscript, vision, locomotion, jackhan-sdu]
summary: "在 MuJoCo 渲染 named 深度相机 depth_cam，经 OpenCV 侧可视化与观测构造，将本体状态与深度拼入 TorchScript（.pt）策略，经 PD 与手柄速度指令驱动 E3 21-DoF 场景；README 声明研究用途许可。"
updated: 2026-11-13
status: complete
related:
  - ./jackhan-feap-mujoco-deployment.md
  - ./jackhan-walke3-e3-ecosystem.md
  - ../concepts/sim2real.md
  - ./mujoco.md
sources:
  - ../../sources/repos/jackhan-feapvision-mujoco-deployment.md
---

# FEAP Vision MuJoCo 部署（深度 + TorchScript）

本仓库在 MuJoCo 中渲染深度相机、拼接本体观测并加载 TorchScript 策略，经 PD 与手柄指令驱动 E3 类 21-DoF 人形。

**定位**：README 以中文描述「**本体观测 + 深度图 + 手柄速度指令**」的 **TorchScript** 部署：依赖 `mujoco`、`torch`、`opencv-python`、`pygame` 等；策略文件放在 `policy/`，配置中可用 `{DEPLOY_DIR}` / `{PACKAGE_ROOT}` 占位符展开路径。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |
| DoF | Degrees of Freedom | 自由度，人形通常 20–50+ 关节 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| MJCF | MuJoCo XML Format | MuJoCo 的模型与场景描述格式 |
| ONNX | Open Neural Network Exchange | 跨框架神经网络模型交换格式 |
| GPU | Graphics Processing Unit | 图形处理器，大规模并行仿真训练的算力基础 |
| CPU | Central Processing Unit | 中央处理器 |

## 核心机制（工程切片）

- **观测链**：深度相机 → 深度预处理模块 → `Observation Builder` → `VisionPolicyRunner`（TorchScript）→ PD 控制 → MuJoCo 步进（README 系统结构图）。
- **场景约束**：要求 MJCF 中存在名为 **`depth_cam`** 的相机；默认场景指向 `e3_21dof/scene_terrain.xml` 一类资源路径（以 README 为准）。
- **推理后端**：与 FEAP ONNX 线不同，本仓强调 **TorchScript `.pt`** 与可选 GPU 版 PyTorch 安装指引。

## 流程总览

```mermaid
flowchart TD
  cam["MuJoCo depth_cam<br/>渲染深度缓冲"]
  pre["vision_depth.py<br/>裁剪 / 尺度 / 范围"]
  obs["Observation Builder<br/>本体 + 深度张量"]
  ts["TorchScript 策略<br/>*.pt"]
  pd["PD / 重力对齐等<br/>utils 层"]
  mj["MuJoCo 人形步进"]
  gp["手柄速度指令"]

  cam --> pre --> obs --> ts --> pd --> mj
  gp --> mj
  mj --> cam
```

## 源码运行时序图

以下时序图基于官方仓库 `JackHan-Sdu/FeapVision_Mujoco_deployment`（`main` 分支）核读的源码与配置梳理：`deploy_mujoco/README.md`、`deploy_mujoco/configs/e3_vision.yaml`、`deploy_mujoco/deploy_mujoco_vision.py`、`deploy_mujoco/vision_deploy/vision_depth.py` 与 `deploy_mujoco/vision_deploy/policy_inference.py`。

```mermaid
sequenceDiagram
    autonumber
    participant MAIN as 视觉部署入口<br/>deploy_mujoco_vision.py
    participant PAD as 手柄输入<br/>utils/gamepad_utils.py
    participant DEPTH as 深度渲染与预处理<br/>vision_depth.py / mujoco.Renderer
    participant POL as 策略推理器<br/>VisionPolicyRunner (policy_inference.py)
    participant MJ as MuJoCo 仿真与视图<br/>MjModel / MjData / viewer

    MAIN->>MAIN: _load_config() 展开 {DEPLOY_DIR} 与 {PACKAGE_ROOT}
    MAIN->>MJ: MjModel.from_xml_path(scene_terrain.xml) 并查找 depth_cam ID
    MAIN->>DEPTH: 初始化 mujoco.Renderer(m, height, width) 与 depth_image_buffer
    MAIN->>POL: VisionPolicyRunner(policy_path) -> torch.jit.load(motion_lstm.pt)
    MAIN->>PAD: init_gamepad(config)
    MAIN->>MJ: mujoco.viewer.launch_passive(m, d)

    loop 主仿真循环（viewer.is_running()）
        opt viewer_utils.reset_requested
            MAIN->>MJ: 重置 qpos/qvel/cmd/action，清空 depth_image_buffer 并调用 policy.reset()
        end
        MAIN->>PAD: 50 Hz 更新 update_cmd_from_gamepad(cmd)
        MAIN->>MJ: pd_control(target_dof_pos, qpos, qvel) -> d.ctrl[:] = tau -> mujoco.mj_step(m, d)
        opt counter % control_decimation == 0 (50 Hz)
            MAIN->>MAIN: _build_obs() 构造 72 维本体与指令观测 obs
            MAIN->>DEPTH: update_depth_cam(depth_renderer, d, depth_cam_id, ...)
            opt cam_update_counter % depth_decimation == 0
                DEPTH->>DEPTH: depth_renderer.update_scene/render() -> rot90 -> cv2.resize -> normalize_depth_image()
                DEPTH-->>MAIN: depth_img [1, 24*32], depth_raw
                MAIN->>MAIN: 滑动窗口更新 depth_image_buffer (2帧拼接) 与 cv2.imshow("depth (raw)")
            end
            MAIN->>POL: policy.infer(obs_t, current_depth_img)
            POL-->>MAIN: action (21-DoF)
            MAIN->>MAIN: target_dof_pos = action * action_scale + default_angles
        end
        MAIN->>MJ: viewer_utils.update_viewer_settings() 与 viewer.sync()
    end
```

- **关键复现路径**：执行 `python deploy_mujoco/deploy_mujoco_vision.py configs/e3_vision.yaml`；内环以 500 Hz 运行 `pd_control` 与 `mj_step`，外环以 50 Hz（`control_decimation=10`）调用 `_build_obs` 与 `VisionPolicyRunner.infer`，其中深度渲染通过 `depth_decimation=5`（10 Hz）经 `update_depth_cam` 更新两帧 `24x32` 深度缓冲。

## 常见误区或局限

- **2026-11-13 维护复核：** [官方 README（30c788c）](https://github.com/JackHan-Sdu/FeapVision_Mujoco_deployment/blob/30c788c06242480bae9f2f61085a89ee4ab58f89/README.md) 仍采用本体观测 + 深度图 + TorchScript，入口为 `deploy_mujoco_vision.py`，不含训练代码；README 末尾标明仅研究与学术用途。本次只核对文档，未运行部署。
- **README 声明**：仓库写明 **仅研究与学术用途**；与通用 OSS 许可不可混为一谈。
- **性能**：README 建议 NVIDIA GPU 以支撑高帧率深度渲染；CPU 路径需自行压测实时性。

## 与其他页面的关系

- **[FEAP MuJoCo 部署](./jackhan-feap-mujoco-deployment.md)**：同一作者的「纯本体 ONNX FEAP」对照组。
- **[Sim2Real](../concepts/sim2real.md)**：视觉策略在仿真渲染深度上闭环，对 sim2real 的感知域随机化提出更高要求。
- [Humanoid Locomotion (人形机器人移动)](../tasks/humanoid-locomotion.md)：在 MuJoCo 地形场景中验证 E3 21-DoF 人形机器人的深度视觉与本体融合移动策略。
- [Robot Simulation Integration · 机器人仿真环境集成](../methods/robot-simulation-integration.md)：将 MJCF 地形场景、depth_cam 深度渲染、TorchScript 策略与 PD 控制器集成为闭环仿真部署环境。

## 参考来源

- [FeapVision_Mujoco_deployment 仓库归档](../../sources/repos/jackhan-feapvision-mujoco-deployment.md)

## 关联页面

- [JackHan-Sdu WalkE3 / HumanoidE3 工具链生态](./jackhan-walke3-e3-ecosystem.md)
- [FEAP MuJoCo 部署（E3 ONNX）](./jackhan-feap-mujoco-deployment.md)
- [Sim2Real](../concepts/sim2real.md)

## 推荐继续阅读

- 上游仓库 README：<https://github.com/JackHan-Sdu/FeapVision_Mujoco_deployment>
