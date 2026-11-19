# Behavior Prompting（real-stanford/behavior_prompting）

- **URL：** <https://github.com/real-stanford/behavior_prompting>
- **关联仓库：** <https://github.com/real-stanford/iPhUMI>（手持数据采集与 `.zarr` 导出）
- **许可：** MIT License
- **语言：** Python（PyTorch / Hydra / RobotMQ / Zarr）
- **配套论文：** [arXiv:2606.30457](https://arxiv.org/abs/2606.30457)
- **项目页：** <https://behavior-prompting.github.io/>
- **入库日期：** 2026-11-19
- **核对范围：** 通读 `README.md`、`LICENSE`、`docs/*.md` 与核心 Python 模块（`train_network/train.py`、`TrainPolicyWorkspace`、`RolloutPolicyWorkspace`、`PairPromptObsEncoder`、`PairPromptTransformerTokenizer`、`DiffusionUnetPolicy`、`deployment/detached_inference.py`）。
- **通读：** 已完成

## 开源范围与已知边界（2026-11-19）

| 项 | 状态 | 详情 |
|----|------|------|
| 训练与离线评估 | **已发布** | `behavior_prompting/train_network/train.py` + `TrainPolicyWorkspace` / `RolloutPolicyWorkspace` |
| 真实部署节点 | **已发布** | `behavior_prompting/deployment/detached_inference.py`（`PolicyInferenceNodeRealEnv` 通过 `robotmq` 提供 `policy_prompt` 与推理服务） |
| OpenPI ($\pi_{0.5}$) 对照脚本 | **已发布** | `docs/libero_openpi.md` 记录对应的 LoRA 微调权重与 LeRobot 数据集 |
| 无线实时 iPhUMI prompt 传输 | **TODO 未发布** | `README.md` 明确标注目前需先从 iPhone 导出并通过 `iPhUMI` 处理成 `.zarr`，再以 `--dataset-path` 传入部署节点 |
| 真机 DrawAnything 程序化脚本 | **未公开** | `docs/drawanything_real.md` 明确说明真机脚本采数生成与 Chamfer 自动评测代码未开源 |

## 仓库核心入口（源码核对）

| 阶段 | 文件 / 类 | 关键机制 |
|------|-----------|----------|
| 训练入口 | `behavior_prompting/train_network/train.py` → `TrainPolicyWorkspace` | Hydra 加载配置后实例化 `DiffusionUnetPolicy` 并调用 `compute_loss` |
| 提示编码 | `behavior_prompting/train_network/model/prompt/prompt_obs_encoder.py` | `PairPromptTransformerTokenizer` 将 prompt 切块并融合高频动作，`PairPromptObsEncoder` 输出上下文条件特征 |
| 动作解码 | `behavior_prompting/train_network/policy/diffusion_unet_policy.py` (`DiffusionUnetPolicy`) | `ConditionalUnet1D` 在条件表征引导下做 16 步去噪生成动作块（`predict_action`） |
| 仿真评测 | `behavior_prompting/train_network/workspace/rollout_policy_workspace.py` (`RolloutPolicyWorkspace`) | 针对 `DrawAnything-Sim` 与 `LIBERO` / `LIBERO-Gen` 执行 rollout 评测 |
| 真机部署 | `behavior_prompting/deployment/detached_inference.py` (`PolicyInferenceNodeRealEnv`) | 启动 `robotmq.RMQServer("policy_server", ...)`，响应 `policy_prompt`（从指定 `.zarr` 加载任务索引）并循环调用 `self.policy.predict_action` |

## 交叉

- 论文归档：[`sources/papers/behavior_prompting_arxiv_2606_30457.md`](../papers/behavior_prompting_arxiv_2606_30457.md)
- 项目页归档：[`sources/sites/behavior-prompting-github-io.md`](../sites/behavior-prompting-github-io.md)
- wiki：[`wiki/entities/paper-behavior-prompting-policy.md`](../../wiki/entities/paper-behavior-prompting-policy.md)
