# LW BENCHHUB TOUR

> 来源归档

- **标题：** LW BENCHHUB TOUR
- **类型：** repo
- **链接：** https://github.com/GimpelZhang/lw_benchhub_tour
- **作者：** GimpelZhang
- **许可：** Apache-2.0
- **Stars：** 3（2026-08-17 核查）
- **默认分支：** `main`（最近推送 2026-07-19）
- **入库日期：** 2026-08-17
- **一句话说明：** 在光轮 LW-BenchHub + Isaac Lab-Arena + LeRobot 上，把 SmolVLA 双臂 Piper 厨房 PnP 跑成 headless 闭环评测，再用 LLM 场景生成 + cuRobo 可达性闸门与自过滤数据飞轮做工程探索。
- **沉淀到 wiki：** [lw-benchhub-tour](../../wiki/entities/lw-benchhub-tour.md)
- **交叉归档：** [LW-BenchHub 官方仓](lw-benchhub.md)、[Lightwheel Platform 项目页](../sites/lightwheel-platform.md)

---

## 步骤 2.5：源码开放核查

| 入口 | 结论 |
|------|------|
| GitHub 仓 | **已开源、可运行**：Apache-2.0；含 Stage 1/2/4 脚本、`stage4_flywheel/`、GitHub Wiki 复现指南 |
| 官方物理底座 | [LightwheelAI/LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub) Apache-2.0；文档 [docs.lightwheel.net/lw_benchhub](https://docs.lightwheel.net/lw_benchhub) |
| 项目页 | [lightwheel.ai/lightwheel-platform](https://lightwheel.ai/lightwheel-platform) 为企业产品叙事；代码入口走 GitHub |
| Hugging Face | 策略 `LightwheelAI/smolvla-double-piper-pnp`（~0.5B）；EnvHub `LightwheelAI/lw_benchhub_env`；训练数据 `LightwheelAI/Lightwheel-Tasks-Double-Piper` |
| 独立项目页 | 本仓无 `*.github.io`；复现入口是仓库 README + [GitHub Wiki](https://github.com/GimpelZhang/lw_benchhub_tour/wiki) |

**运行边界：** 需要 Isaac Sim 5.1 + Isaac Lab 2.3.2 + IsaacLab-Arena `release/0.1.1` + 大显存 GPU（作者环境 A800 40GB headless）。不是 `pip install` 即可复现的轻量仓。

---

## 核心定位

把 NVIDIA **EnvHub** 模式（`lerobot-eval --env.type=isaaclab_arena --env.hub_path=...`）接到光轮厨房任务与 **DoublePiper-Abs** 双臂上，系统跑通三条工程闭环：

1. **Stage 1** — SmolVLA 高频物理闭环评测基线（`L90K1PutTheBlackBowlOnThePlate`，10 episode **40%**）。
2. **Stage 2** — LLM 改写 YML 场景 + **进程内** Isaac Lab boot + cuRobo IK **可达性闸门**。
3. **Stage 4** — 课程场景 + 失败诊断 + SmolVLA **自过滤** 导出 LeRobotDataset（scripted cuRobo PnP **未**产出可微调成功轨迹）。

Stage 3 被作者丢弃（效果不好）。

---

## 仓库结构（作者侧入口，不含 vendored IsaacLab / lerobot 子树）

| 路径 | 角色 |
|------|------|
| `generate_scenes_with_live_reach.py` | Stage 2：LLM 生成 + live cuRobo reach gate |
| `validate_scene_objects_reach.py` | 场景物体左右臂 IK 可达性校验 |
| `verify_stage2.py` | Stage 2 交付物核验 |
| `stage4_flywheel/` | 飞轮脚本、双臂 URDF 生成、HDF5→LeRobot 导出 |
| `piper_curobo.yml` | 单臂 cuRobo 配置入口 |
| `deepseek_v4_pro.py` | DeepSeek-v4-pro 调用（防降级回读） |
| `Dockerfile.eval` / `run_eval_docker.sh` | 评测容器 |
| `CLAUDE.md` | 交接手册：numpy 锁定、健康自检、ABI 坑 |
| GitHub Wiki | `Complete_Stage_1` / `_2` / `_4`、`LW_Benchhub_Interface` |

---

## 实测栈（Wiki Stage 1，作者环境）

| 组件 | 版本 |
|------|------|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.2 |
| IsaacLab-Arena | `release/0.1.1` |
| LeRobot | 0.5.1 |
| LW-BenchHub | 0.1.0 editable |
| Python / numpy | 3.11.15 / **1.26.0 锁定** |
| 任务 | `L90K1PutTheBlackBowlOnThePlate`（RoboCasa/LIBERO 厨房） |
| 本体 | DoublePiper-Abs：观测 16-D，动作 12-D（左右臂各 5-DoF + 二值夹爪，跳过 `joint4`） |

---

## 对 wiki 的映射

- 实体：[lw-benchhub-tour](../../wiki/entities/lw-benchhub-tour.md)
- 栈对照：[LeRobot](../../wiki/entities/lerobot.md)、[Isaac Lab](../../wiki/entities/isaac-lab.md)、[cuRobo](../../wiki/entities/curobo.md)
- 任务：[Bimanual Manipulation](../../wiki/tasks/bimanual-manipulation.md)
- 部署对照：[VLA 真机部署指南](../../wiki/queries/vla-deployment-guide.md)（本仓是仿真闭环，不是真机）
