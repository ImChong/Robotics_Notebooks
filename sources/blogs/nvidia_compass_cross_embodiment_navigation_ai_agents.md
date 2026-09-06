# How to Train a Cross-Embodiment Robot Navigation Policy with AI Agents

> 来源归档

- **标题：** How to Train a Cross-Embodiment Robot Navigation Policy with AI Agents
- **类型：** blog / 官方教程
- **URL：** https://developer.nvidia.com/blog/how-to-train-a-cross-embodiment-robot-navigation-policy-with-ai-agents/
- **发布日期：** 2026-08-26
- **作者：** Yan Chang, Mihir Acharya, Wei Liu, Katie Washabaugh, Aishwarya Singh
- **入库日期：** 2026-09-06
- **一句话说明：** COMPASS 的 **agent-driven 工作流教程**：开发者定义机器人/场景/导航目标，编码 agent 用仓库 skills 完成依赖校验、资产准备、smoke test、残差 RL 训练、checkpoint 评测与 ROS2 集成；参考平台为 **Spot + 内置仓库 / SAGE-10K / NuRec** 场景。
- **沉淀到 wiki：** [`wiki/entities/compass.md`](../../wiki/entities/compass.md)

---

## 核心叙事（2026-09-06 抓取）

### 问题

把导航能力迁到新机器人或新场景，往往需要重新收集数据、建仿真资产、接机器人接口、训练、诊断与评测——对每个「机器人×场景」重复一遍，成本高且难复现。

### Agent 工作流

开发者定义 **机器人、场景来源、导航目标**；**编码 agent**（教程用 **Codex**，Claude Code 用 `/compass`）通过仓库 **skills** 自动完成：

- 依赖与栈校验（钉死仓库 revision、容器、GPU、Isaac Lab/Sim、资产与 X-Mobility ckpt）
- 场景准备与 occupancy map
- **单环境 smoke test**（人工审批门）
- 残差 RL 训练与 checkpoint 管理
- 失败诊断（`$compass-doctor`）
- 匹配条件下的 base vs residual 评测

**人工审批门：** 场景接受、单环境 smoke test、checkpoint 晋升——每阶段需可审查证据。

### 参考路径（Spot 四足）

| 场景来源 | 说明 |
|----------|------|
| **内置仓库** `combined_multi_rack` | 最快可复现基线；机器人/场景/占据图已注册 |
| **SAGE-10K** | 1 万室内生成场景；需 USD 转换、注册、占据图与两次人工审批 |
| **Omniverse NuRec** | 可选 Real2Sim 捕获环境；教程主流程仍以 SAGE-10K 贯通 |

### 训练命令示例（残差 RL）

```bash
python run.py \
  -c configs/train_config.gin \
  -o ./outputs/spot_combined_multi_rack \
  -b ./assets/x_mobility.ckpt \
  --enable_cameras \
  --embodiment spot \
  --environment combined_multi_rack
```

- Smoke test 用 `--num_envs 1`；全量训练按 GPU 显存调 `num_envs`。
- 支持多 GPU 分布式；须保存配置、revision、日志与 checkpoint 清单。

### 评测指标

标准 COMPASS 评测：**goal-reached rate、fall-down rate、travel time**；晋升 checkpoint 前需 base 与 residual 在匹配种子/目标/初始态下对比。

### 部署（ROS 2）

- `compass_inference`：前视 RGB + 导航目标/路线 + 里程计速度 → `/cmd_vel`（线速度/角速度）。
- **cuVSLAM** 可选：GPS 受限时提供相机里程计；非训练组件，独立 ROS2 节点对接 `/chassis/odom` 与 TF。
- 导出 ONNX/JIT/TensorRT 与真机硬件需单独验证；教程止于 checkpoint 评测。

### 新机器人

未注册机器人用 **`$compass-newembodiment`** skill 扩展。

## 对 wiki 的映射

- [compass](../../wiki/entities/compass.md)
- [compass 仓库归档](../repos/compass.md)
- [nvidia-brev](../../wiki/entities/nvidia-brev.md) — 无本地 GPU 时的云开发环境
