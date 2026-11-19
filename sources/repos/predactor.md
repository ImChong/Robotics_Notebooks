# PredActor（官方代码仓：MasterYip/PredActor）

> 来源归档（最近复核：2026-11-19）

- **类型：** repo / humanoid / evaluation / collection / bc-training / deployment
- **链接：** <https://github.com/MasterYip/PredActor>
- **项目页：** <https://masteryip.github.io/predactor.github.io/> — [团队与联系](https://masteryip.github.io/predactor.github.io/#people)
- **论文：** [arXiv:2609.24840](https://arxiv.org/abs/2609.24840) — [`sources/papers/predactor_arxiv_2609_24840.md`](../papers/predactor_arxiv_2609_24840.md)
- **公开评测入口：** `predactor-eval`（`PredActor/cond_eval.py`）
- **许可：** MIT（仓库）；机器人资产与第三方依赖仍受各自上游许可约束
- **Python：** 3.10（`uv sync --locked` 锁定环境）
- **入库日期：** 2026-09-24
- **最近复核：** 2026-11-19
- **核对范围：** `README.md`、`pyproject.toml`、`scripts/hf_download.py`、`PredActor/cond_eval.py`、`PredActor/train.py`、`PredActor/collection/collect_dataset.sh`、`PredActor/diffusion_policy/{agent/bc_agent.py,modules/diffuse_cloc.py,modules/diffuse_cprdp.py,config_files/g1prdp_cond_diffuse.yaml,config_files/defaults/env_runner.yaml}`、`PredActor/diffusion_deploy/doc/DEPLOYMENT.md`、`docs/training.md`
- **通读：** 已完成
- **一句话说明：** PredActor 官方仓库已从早先仅含 MuJoCo 评测升级为覆盖**评测、数据采集与语义染色、离线 BC 训练与 LibTorch C++ G1 部署**的主体开源仓库，当前仅 DAgger 聚合模块仍未开放。

## 当前公开范围（2026-11-19 源码核对）

| 能力 | 状态 | 入口与源码依据 |
|------|------|----------------|
| MuJoCo Web UI 评测 | **已开放** (`[x]`) | `predactor-eval` → `PredActor/cond_eval.py`；支持文本条件、摇杆、鼠标推扰与点导航 WBG |
| 数据采集与 MotionCLIP 染色 | **已开放** (`[x]`) | `PredActor/collection/collect_dataset.sh`（从 RLTracker + 152 重定向动作采原始 zarr）及语义染色管线 |
| 离线 BC 训练 | **已开放** (`[x]`) | `PredActor/train.py`（支持 `g1prdp_cond_diffuse.yaml` / `g1prdp_diffuse.yaml`，强行关闭训练期 guidance）与 `PredActor/scripts/training_smoke.py` |
| G1 机载 / Sim2Sim 部署 | **已开放** (`[x]`) | `PredActor/diffusion_deploy/`（`tools/export_policy.py` 导出 TorchScript + C++ `build.sh` 编译 LibTorch / Unitree DDS 运行时，支持 native & `jetson-jp5-torch21`） |
| 评测与部署工件 | **已开放** | [PredActor_Artifacts](https://huggingface.co/MasterYip/PredActor_Artifacts)（经 `scripts/hf_download.py` 拉取并核对 SHA-256） |
| DAgger 策略精炼 | **未开放** (`[ ]`) | README Release Checklist 仍未勾选；`cond_eval.py` 留有 `_is_dagger_checkpoint` 兼容接口但 `dagger/` 训练源码未公开 |

## 核心源码模块（对齐时序图）

| 路径 / 符号 | 角色 | 关键实现细节 |
|-------------|------|--------------|
| `PredActor/train.py` (`main`) | BC 训练入口 | 解析 `--cfg`（如 `g1prdp_cond_diffuse.yaml`）与覆写参数，强行将 `enable_root_pos_guide` 及嵌套 guidance `enabled` 设为 `false`，经 Hydra 实例化 `OfflineTrainer` 并调用 `trainer.train()` |
| `PredActor/diffusion_policy/agent/bc_agent.py` (`BCAgent`) | 策略代理封装 | 组合 `CondCompressor`、`PerTermConditionDropout`/`AdaptiveConditionDropout`、`TaskMaskGenerator` 与 `JointDiffusionActor`/`DiffuseCPRDP`，统一归一化（`normalizer`）并在 `compute_loss` 调用 `actor.p_losses`、在推理调用 `actor.act` |
| `PredActor/diffusion_policy/modules/diffuse_cloc.py` & `diffuse_cprdp.py` (`DiffuseCLoC`, `DiffuseCPRDP`) | 联合扩散主干 | 维护 FIFO 滚动去噪缓冲区、状态重点投影（`state_emphasis`）、异步动作/状态调度（`action_schedule`/`state_schedule`）、`NS-DDIM` 与 `WholeBodyGuidance`（及 `RootPosGuideProvider`） |
| `PredActor/cond_eval.py` (`main`, `_load_eval_agent`) | 闭环仿真评测 | 加载 checkpoint 与 `CompositeCondProvider`（`CLIPTeleop` / `CLIPInterp`），启动 `EvaluationWebSession` 与 `EnvRunner` 在 50 Hz（`control_dt: 0.02`）闭环运行 |
| `PredActor/diffusion_deploy/tools/export_policy.py` & `doc/DEPLOYMENT.md` | C++ 部署导出 | 导出 TorchScript 主干、配置与 SHA256SUMS，用于 Unitree MuJoCo 环回 DDS、有线 G1 或 Jetson 机载部署 |

## 复现边界提示

- `cond_eval.py` 与 `diffusion_deploy` 使用 TorchScript + LibTorch FP32（`use_fk: false` 对应 PRDP 剖面），并非 ONNX/TensorRT。
- 因 **DAgger 模块未公开**，开源仓库目前支持复现“动作库采集 → MotionCLIP 染色 → 离线 BC 训练 → 仿真与 G1 部署”，尚未提供论文中闭环 DAgger 在线迭代纠偏阶段。

## 对 wiki 的映射

- [PredActor 论文实体](../../wiki/entities/paper-predactor.md)
- [评测 checkpoint 归档](predactor-artifacts.md)
- [项目页归档](../sites/predactor-masteryip-github-io.md)
- [论文来源归档](../papers/predactor_arxiv_2609_24840.md)
