# Interactive-Active-AI/OneWM（官方代码仓）

> 来源归档（ingest）

- **标题：** OneWM: Can World and Action Modeling Benefit Each Other under Unified Causal Training?
- **类型：** repo
- **URL：** <https://github.com/Interactive-Active-AI/OneWM>
- **默认分支：** main
- **许可证：** Apache-2.0（仓库根目录 LICENSE）
- **配套论文：** [arXiv:2610.21095](../papers/onewm_arxiv_2610_21095.md)
- **项目页：** <https://selen-suyue.github.io/OneWMNet/> — [项目页归档](../sites/onewm-project.md)
- **入库日期：** 2026-10-19

## 一句话摘要

OneWM 仓库公开其因果视频—动作 MoT 的训练、推理和评测实现，包含 LIBERO、RoboCasa365 和双 Franka 方案；用户仍需自行准备数据、基础 Wan2.2 权重与训练 checkpoint。

## 代码入口

| 用途 | 仓库入口 | 说明 |
|---|---|---|
| 训练 | [`scripts/train.py`](https://github.com/Interactive-Active-AI/OneWM/blob/main/scripts/train.py)、`onewm.training.train` | Accelerate 启动；支持 `configs/libero.yaml`、`configs/robocasa365.yaml`、`configs/franka.yaml`。 |
| 模型初始化 | `onewm.tools.preprocess_action_dit_init` | 将 Wan transformer 权重插值到 ActionDiT 专家，生成初始化文件；这不是训练好的策略权重。 |
| WAM 推理 | `onewm.tools.infer --mode wam` | 从观察与任务指令生成归一化动作块。 |
| WM 推理 | `onewm.tools.infer --mode wm` | 给定动作块，预测未来视频帧。 |
| 在线评测 | `deploy/libero/`、`deploy/robocasa365/` | 分别通过 policy server 与仿真客户端交互评测。 |
| Franka 适配 | `deploy/franka/` | 提供离线策略适配器；硬件客户端负责同步、单位、控制频率与安全限制。 |
| 想象 rollout | `deploy/libero/endless_dream.py` | 交替预测动作与其视觉后果，把最后生成帧作为下一轮输入；默认不执行预测动作。 |

## 可复现性与边界

- **软件依赖：** README 给出 Linux、NVIDIA CUDA、PyTorch 2.6 / Python 3.11、Accelerate 等环境设置，并从 Hugging Face 下载 Wan2.2-TI2V-5B 初始化骨干。
- **数据：** LIBERO、RoboCasa365 与 Franka 各自有配置和准备说明；仓库不附数据集。RoboCasa365 仿真资产也须另外下载，Franka 录制由使用者提供。
- **检查点：** 仓库示例把权重路径作为用户参数，不提供可直接下载的 OneWM checkpoint；预处理 ActionDiT 权重不等于训练 checkpoint。
- **真实机器人：** Franka adapter 返回动作目标，不直接连接机器人控制器；实际硬件执行需要自有控制客户端与受监督安全流程。
- **结果复现：** 对齐数据版本、归一化统计、recipe、text cache 与评估协议；论文 benchmark 数字不应被解读为任意本体或环境的保证。

## 关联资料

- 论文：[sources/papers/onewm_arxiv_2610_21095.md](../papers/onewm_arxiv_2610_21095.md)
- 项目页：[sources/sites/onewm-project.md](../sites/onewm-project.md)
- 论文实体：[wiki/entities/paper-onewm-unified-causal-training.md](../../wiki/entities/paper-onewm-unified-causal-training.md)
