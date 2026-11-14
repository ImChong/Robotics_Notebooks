---
type: source
source_type: repo
title: "Roboparty/human-humanoid-tools (hhtools) 源码与双后端管线核查"
urls:
  - "https://github.com/Roboparty/human-humanoid-tools"
  - "https://github.com/Roboparty/human-humanoid-tools/releases/tag/0.1.0(beta)"
created: 2026-11-14
updated: 2026-11-14
tags: [humanoid, motion-retargeting, open-source, roboparty, newton-physics]
---

# `Roboparty/human-humanoid-tools`（`hhtools`）源码核查

- **官方仓库：** <https://github.com/Roboparty/human-humanoid-tools>
- **核对提交：** `13b6dd3024193aeb24eb4bf7ea5c73a975a38f5b`（`main` 分支）
- **核对范围：** 核查 `README.md`、`pyproject.toml`、`framework.md`、`configs/builtin-assets.json`、`hhtools/cli/main.py`、`hhtools/cli/retarget.py`、`hhtools/application/retarget.py`、`hhtools/retarget/newton_basic/pipeline.py`、`hhtools/retarget/interaction_mesh/pipeline.py`、`hhtools/retarget/robot_to_robot.py`、`hhtools/web/server/factory.py` 与 `hhtools/mcp/server.py`
- **通读：** 已完成

## 一、开源开放状态核查（步骤 2.5）

| 资源层级 | 开放状态 | 说明与限制 |
|---|---|---|
| 核心管线与双后端代码（`hhtools`） | **已开源（Apache-2.0）** | `pyproject.toml` 声明 Python `>= 3.12`，CLI / WebUI / MCP / Newton IK / Interaction-Mesh / R2R 源码全量在 GitHub 公开；`newton_basic` 模块标注适配自 NVIDIA `SOMA-Retargeter`（Apache-2.0），`interaction_mesh` 架构对标 `holosoma` `InteractionMeshRetargeter` |
| 内置机器人与示例动作清单 | **随仓配置（许可独立声明）** | `configs/builtin-assets.json` 声明 `7` 款内置机器人（ROBOTO 01、Unitree G1、ROBOTO_ORIGIN (RPO)、AgiBot X2、Asimov 1、Fourier GR-2、Berkeley Humanoid Lite）与 AMASS / GLB / LAFAN / meshmimic / MOCAP / Motion-X / OmniContact / OMOMO / PARC MS / PHUMA / SOMA / Xsens 示例片段，明确标注随仓存在不代替发布前许可审查 |
| SMPL / SMPL-H / SMPL-X 身体模型权重 | **未随仓打包（需自行申请）** | 权重存于本地 `~/.cache/hhtools/body_models/`，需遵循 MPI 非商业授权条款自行下载配置 |
| 可选 GPU 物理与视频动捕依赖 | **外部包管理** | Newton IK 后端依赖 NVIDIA `newton` / `warp`（CUDA Graph 加速与 GPU 内核首次编译）；视频重建（GVHMR）与地形求解（OSQP）为可选依赖 |

## 二、核心模块与实读入口总结

1. **四类执行入口：**
   - **CLI（`hhtools.cli.main:app`）：** Typer 子命令体系涵盖 `convert`、`import`、`bodymodel`、`doctor`、`robot`、`retarget`、`web`、`agent`。其中 `hhtools/cli/retarget.py` 提供 `retarget run`（Newton IK NPZ 批量管线）以及 `retarget interaction-mesh precompute` / `retarget interaction-mesh run`（支持 NPZ、BVH、CSV、OMOMO `.pkl`、PARC `.pkl`、meshmimic `.npy` 与 OmniContact 捕捉目录）。
   - **WebUI（`hhtools.web.server.factory:_mount_web_app`）：** 本地 FastAPI 服务 + HTML/three.js 渲染（`uv run hhtools web`，默认 `127.0.0.1:8009`），统一复用 `ApplicationRuntime`（任务调度器 `JobScheduler`、上传存储 `UploadStore`、会话状态 `SessionState`），旧版 Viser 查看器已在架构演进中退场（`hhtools/viewer/` 仅作兼容重导出）。
   - **Desktop（`hhtools-desktop`）：** Electron + sidecar 桌面包装，预览版 Release 为 `0.1.0(beta)`（2026-09-24）。
   - **Agent / MCP（`hhtools/mcp/server.py`）：** 通过 `hhtools-mcp` 启动本地 stdio MCP v2 服务器（与 WebUI 通过租约 `RuntimeLeaseError` 互斥共享同一 `--save-dir`），暴露资产检索/注册、标定提案与预览（`propose_calibration` / `propose_r2r_calibration`）、预检（`preflight_retarget` / `preflight_r2r` / `preflight_batch`）、任务执行与工件导出工具。
2. **双重定向求解后端与 R2R：**
   - **Backend A（`NewtonBasicPipeline`）：** `HumanToRobotScaler` → `FeetStabilizer` → Warp `newton.ik` 逐帧求解（支持 `wp.ScopedCapture` 记录 CUDA Graph 并在多夹批处理中构建 `num_envs=N` 并行模型）→ `JointLimitClamper` → 速度限幅，输出 `RetargetedMotion`。
   - **Backend B（`InteractionMeshPipeline`）：** 基于身高比均匀缩放人体关节与物体交互场景 → 仅对 `ik_map` 映射关节构建 Delaunay 四面体与目标拉普拉斯坐标 $\delta^*_f = L V_f$ → `iterate_mpc` 调用 MuJoCo `mj_forward` 与雅可比 `J_V` 进行信赖域 MPC/SQP 求解（L-BFGS-B / dense KKT 或 OSQP 地形约束，`OSQP_FALLBACK_TRUST_SHRINK = 0.25`）。
   - **Robot-to-Robot（`hhtools/retarget/robot_to_robot.py`）：** 读取源机器人 `csv/pkl/npz` 轨迹并执行 URDF 前向运动学提取规范关键点 `Motion`，结合源零位参考姿态标定后，分别派发至 `InteractionMeshPipeline` 或 `NewtonBasicPipeline`（默认可选）迁移至目标机器人。

## 参考链接

- [Roboparty/human-humanoid-tools 仓库](https://github.com/Roboparty/human-humanoid-tools)
- [HHTools 0.1.0(beta) Release](https://github.com/Roboparty/human-humanoid-tools/releases/tag/0.1.0(beta))
