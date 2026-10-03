# Jbo-Wang/discrete_forcing

> 来源归档（repo）

- **名称：** discrete_forcing
- **类型：** repo / vla / flow-matching / manipulation / training / evaluation
- **URL：** <https://github.com/Jbo-Wang/discrete_forcing>
- **论文：** [arXiv:2609.39526](../papers/discrete_forcing_arxiv_2609_39526.md)
- **项目页：** <https://discrete-forcing.github.io/> — [开放状态核查](../sites/discrete-forcing-github-io.md)
- **机构：** 香港科技大学（广州）、华南理工大学、中国科学技术大学、西湖大学、浙江大学、清华大学
- **许可证：** MIT（仓库声明；LICENSE 保留 StarVLA Team 版权与上游署名要求）
- **入库日期：** 2026-10-03
- **一句话说明：** 官方实现提供 StarVLA 基础上的 Discrete Forcing action expert，以及 LIBERO 训练、policy serving 和仿真评测流程。

## 运行入口（README）

| 步骤 | 脚本 / 模块 | 作用 |
|------|-------------|------|
| 数据准备 | `bash prepare_data.sh` | 下载四套 LIBERO LeRobot 数据集并放置 modality 配置 |
| 训练 | `bash train.sh` | 通过 Accelerate 启动 StarVLA 训练入口与 LIBERO 配置 |
| 评测 | `bash examples/LIBERO/eval_files/eval_four_suites.sh --checkpoint ...` | 启动 policy server，在四套件运行评测并汇总结果 |
| 检查 | `python check_install.py` | 检查环境；可选初始化模型 / 数据加载器 |

## 关键路径

| 路径 | 作用 |
|------|------|
| `starVLA/` | 共享视觉语言 backbone、action expert 与训练实现 |
| `examples/LIBERO/train_files/libero_method.yaml` | LIBERO 联合训练配置 |
| `examples/LIBERO/eval_files/` | policy server 与四套件评测脚本 |
| `docs/LIBERO.md` | 环境、数据准备、训练与评测说明 |

## 开放边界（2026-10-03）

- **已开源：** LIBERO 训练与评测代码；项目页链接到本仓库。
- **需另行准备：** LIBERO 数据、Qwen3.5-0.8B VLM 与运行 checkpoint；仓库 README 明确没有包含预训练权重。
- **未完成：** README 将 RoboTwin 训练和评测支持列为 TODO；不能据论文结果推断该套件已能由当前仓库直接复现。
- **部署边界：** README 的评估客户端一次执行 8 个动作后重新规划；论文所示两次 NFE 是动作专家内部生成步骤数，不是机器人电机控制频率。
