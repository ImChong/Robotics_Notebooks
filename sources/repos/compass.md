# NVlabs/COMPASS

> 来源归档

- **标题：** COMPASS: Cross-embOdiment Mobility Policy via ResiduAl RL and Skill Synthesis
- **类型：** repo
- **来源：** NVIDIA Research（NVlabs）
- **链接：** https://github.com/NVlabs/COMPASS
- **项目页：** https://nvlabs.github.io/COMPASS/
- **文档：** https://nvlabs.github.io/COMPASS/docs/
- **论文：** https://arxiv.org/abs/2502.16372
- **许可证：** Apache 2.0
- **入库日期：** 2026-09-06
- **一句话说明：** 跨具身移动策略框架：单具身 IL（X-Mobility 世界模型+策略）→ 残差 RL 专精 → 蒸馏为具身嵌入条件通才策略；Docker + Isaac Lab 3.0 beta，含 agent skills、ROS2 部署与 GR00T 后训数据。
- **代码：** https://github.com/NVlabs/COMPASS（**已开源** Apache 2.0）
- **沉淀到 wiki：** [`wiki/entities/compass.md`](../../wiki/entities/compass.md)

---

## 栈与依赖（README，2026-09-06）

| 组件 | 版本 |
|------|------|
| Isaac Lab | 3.0.0-beta1 |
| Isaac Sim | 4.5.0 |
| 平台 | Linux x86_64 |
| 运行时 | Docker + NVIDIA Container Toolkit + GPU |

## 三阶段方法

1. **模仿学习（IL）：** 在单具身上用教师策略预训练 **世界模型 + 移动策略**（基座为 **NVIDIA X-Mobility** 预训练权重）。
2. **残差 RL：** 在基座动作上加 **残差专精策略**，按机器人/场景修正，无需从零学导航。
3. **策略蒸馏：** 多专精策略蒸馏为 **单一通才策略**，以 **具身嵌入向量** 条件化。

## Quick start（官方）

```bash
git clone https://github.com/NVlabs/COMPASS.git && cd COMPASS
export HF_TOKEN=hf_xxx   # 需访问 gated HF 仓 nvidia/COMPASS、nvidia/X-Mobility
./docker/run.sh assets    # USD + x_mobility.ckpt → ./assets/
./docker/run.sh build
source ./docker/activate
python run.py -c configs/train_config.gin -o /tmp/out -b ./assets/x_mobility.ckpt \
  --enable_cameras --num_envs 1 --visualizer kit
```

- `python` 为容器内 shim；宿主机编辑代码通过 bind-mount 热重载。
- Handbook 还覆盖：训练/蒸馏/导出、ROS2 部署、OSMO 云提交、GR00T 后训、agentic skills、自动 occupancy map 等。

## Agent skills（仓库内）

| Skill | 用途 |
|-------|------|
| `$compass` | 主工作流：环境校验、场景准备、smoke test、训练、评测 |
| `$compass-doctor` | 只读健康检查与失败诊断 |
| `$compass-newembodiment` | 新机器人注册与集成 |
| `$cuvslam-onboard` / `$cuvslam-troubleshoot` | 可选 cuVSLAM 里程计对接 |

Codex：symlink `.claude/skills/*` → `.agents/skills/`；Claude Code 用 `/compass`。

## 评测与真机

- 通才策略相对 IL 基座约 **5×** 成功率、**3×** 更低行程时间（项目页）。
- **零样本 Sim2Real**：Carter、G1 等真机部署叙事。
- 开放词汇导航：与 **Locate3D** 集成；GR00T 可用 COMPASS 蒸馏数据集做导航后训。

## 对 wiki 的映射

- 实体页：[compass](../../wiki/entities/compass.md)
- 官方教程博客：[nvidia_compass_cross_embodiment_navigation_ai_agents](../blogs/nvidia_compass_cross_embodiment_navigation_ai_agents.md)
- 云 GPU：[nvidia-brev](../../wiki/entities/nvidia-brev.md)
- 跨具身：[cross-embodiment-transfer-strategy](../../wiki/queries/cross-embodiment-transfer-strategy.md)
- [isaac-lab](../../wiki/entities/isaac-lab.md)、[isaac-gr00t](../../wiki/entities/isaac-gr00t.md)
