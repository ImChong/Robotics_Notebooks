# MSFlow 官方代码仓（LY Corporation）

> 来源归档（最近复核：2026-10-10）

- **仓库：** <https://github.com/lycorp-jp/MSFlow>
- **项目页：** <https://yu1ut.com/MSFlow-HP/> — [项目页来源归档](../sites/motionspaceflow-yu1ut.md)
- **论文：** [arXiv:2609.34190](https://arxiv.org/abs/2609.34190) — [论文来源归档](../papers/motionspaceflow_arxiv_2609_34190.md)
- **Hugging Face：** <https://huggingface.co/ly-corporation/MSFlow>
- **许可：** CC0 1.0（README）；第三方软件许可见仓库 NOTICE
- **状态：** 官方 README 明确说明这是**临时开放**的仓库，可能随时改为只读或私有；不接受代码贡献
- **运行环境：** 使用 `uv sync` 管理 Python 依赖

## 可复现入口（README）

准备依赖、模型、HumanML3D/SnapMoGen 数据及评估资源后：

| 任务 | 官方命令 |
|------|----------|
| 263D 文本→动作 demo | `uv run python -m sample.demo_msflow_263 name=MMDiT_pretrained` |
| XYZ 文本→动作 demo | `uv run python -m sample.demo_msflow_xyz name=MMDiT_xyz_pretrained` |
| 关节约束 XYZ 采样 | `bash sample/demo_joint_control.sh` |
| 训练 263D | `uv run python -m train.train_msflow_263 name=<exp_name>` |
| 训练 XYZ | `uv run python -m train.train_msflow_xyz name=<exp_name>` |
| 评测 263D / XYZ | `uv run python -m eval.eval_msflow_263 name=<exp_name>` / `uv run python -m eval.eval_msflow_xyz name=<exp_name>` |

预训练权重由 [Hugging Face 仓](https://huggingface.co/ly-corporation/MSFlow)提供；评估依赖建议按 README 从 [MARDM](https://github.com/neu-vi/MARDM) 取得 GloVe 和 t2m evaluators。数据需分别按 [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 与 [SnapMoGen](https://huggingface.co/datasets/Ericguo5513/SnapMoGen) 的上游说明准备。

## 复现边界

- 仓库公开了 demo、训练、评测代码和预训练模型下载方式，属于可运行研究实现。
- 官方 README 标注临时开放；长期可用性不保证。归档与复现时记录仓库 commit，避免把可访问状态视为永久承诺。
- CC0 声明适用于 LY Corporation 所有的代码与材料；第三方软件、数据集、评估器及资产仍需各自遵循上游许可。
- 代码生成的是人体运动序列，不包含机器人 retarget、碰撞/接触求解或低层控制器。

## 对 wiki 的映射

- [MotionSpaceFlow 项目实体](../../wiki/entities/paper-motionspaceflow.md)
- [项目页归档](../sites/motionspaceflow-yu1ut.md)
- [论文归档](../papers/motionspaceflow_arxiv_2609_34190.md)
