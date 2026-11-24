# μ₀（Yoonkyo/mu0）

- **标题：** μ₀ — 3D Interaction-Trace World Model
- **类型：** repo
- **链接：** <https://github.com/Yoonkyo/mu0>
- **项目页：** <https://mu0-wm.github.io/>
- **论文：** [arXiv:2606.13769](https://arxiv.org/abs/2606.13769)
- **入库日期：** 2026-10-02
- **一句话说明：** μ₀ 官方实现：3D 轨迹世界模型训练、评测与下游动作模块接口；数据管线另见 [TraceExtract](traceextract-mu0.md)。

## 开源结论（步骤 2.5）

- **已开源** — 以 README 安装/训练/评测步骤为准；Apache-2.0。2026-11-24 已核查以下官方入口。

## 运行入口

- [评测指南](https://github.com/Yoonkyo/mu0/blob/main/docs/release/EVALUATION.md)：`src/lerobot/scripts/lerobot_predict_trace_mu0_image_only.py` 加载 `SmolVLAPolicy`、`load_trace_stats` 与 `TraceDataset`，调用 `policy.predict_trace`；输出 overlay 和有真值时的指标。
- [训练指南](https://github.com/Yoonkyo/mu0/blob/main/docs/release/TRAINING.md) 与 [策略指南](https://github.com/Yoonkyo/mu0/blob/main/docs/release/TRAINING_POLICY.md)：`train_policy_mu0_atomic8.sh` / `eval_policy_mu0_atomic8.sh`；后者调用 `lerobot_eval_policy_mu0.py`，进行 RoboCasa rollout。
- [模型权重、统计与测试集](https://huggingface.co/furonghuang-lab/mu0)；[策略权重](https://huggingface.co/furonghuang-lab/mu0-policy)。
- 未实际运行；完整预训练语料与 UR3 实机部署代码尚未核实。

## 交叉

- [`wiki/entities/paper-mu0-wm.md`](../../wiki/entities/paper-mu0-wm.md)
- [`sources/papers/mu0_wm_arxiv_2606_13769.md`](../papers/mu0_wm_arxiv_2606_13769.md)
