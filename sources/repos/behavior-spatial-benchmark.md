# BEHAVIOR-Spatial 官方评测实现与源码入口

- **仓库：** https://github.com/arielchen07/BEHAVIOR-Spatial-benchmark
- **核对提交：** `e9c96b96375d2a0b50b3f212e81aa8f101e65442`（2026-11-07 抓取时 main）
- **论文：** https://arxiv.org/abs/2610.34051
- **项目：** https://arielchen07.github.io/behavior-spatial/
- **数据：** https://huggingface.co/datasets/behavior-spatial-benchmark/BEHAVIOR-Spatial
- **开源：** Apache-2.0（项目 LICENSE；NOTICE 说明派生代码）；**仅评测代码，不含论文策略训练实现与 checkpoint**。
- **核对范围：** 仓库 README 全文、`docs/protocol.md`、`bench/sim_eval/run_eval.py` 入口/实例选取与分组、`bench/sim_eval/eval_subtask.py` 中的 BDDL/OmniGibson evaluator 入口、`bench/sim_eval/score.py` 计分路径、`bench/vqa_ir/score_vqa.py`、`score_ir.py`、`examples/hold_policy_server.py` 的协议及公开运行路径；未执行仿真或下载约 183GB 数据。
- **通读：** 未完成

## 已核实的调用边界

1. **VQA：** `python bench/vqa_ir/score_vqa.py --data $DATA/vqa --predictions answers.json`，按问题 ID 与选择题真值比对。
2. **IR：** `python bench/vqa_ir/score_ir.py --data $DATA/ir --predictions points/`，预测 `x_px/y_px`、`x_1000/y_1000` 或原始模型输出；按每任务 strict/tolerant 分数汇总。
3. **Action：** 在独立 policy 环境运行 `python examples/hold_policy_server.py --port 8000`（演示策略仅原地保持，**不能代表论文 baseline**），仿真环境使用 `python bench/sim_eval/run_eval.py --eval-root $DATA/action/eval/manipulation --slice held_out --host localhost --port 8000 --out runs/my_method`。
4. `run_eval.py` 依 `protocol.yaml` 解析任务、评测分片、实例 JSON+BDDL+TRO，并调用 `eval_subtask.py` 或 `eval_subtask_nav.py` 进行 OmniGibson rollout。policy server 以 msgpack/websocket 收到 reset 指令及 RGB/本体观测，回复 `float32[23]` 动作。
5. `python bench/sim_eval/score.py --results runs/my_method` 扫描结果 JSON，汇总操作 strict/soft/wrong-target 与导航 success；`--with-ir` 标记的使用仿真真值 IR **单独汇报 privileged block**，不得混入常规 policy 结果。

## 环境限制与注意事项

- OmniGibson 3.7.2 覆盖 2025 场景；四个餐厅导航任务要求 OmniGibson 3.9.2 / Isaac Sim 5.1。
- 官方示例策略用于验证 harness 连通，并不等于论文中的 Qwen3.5-4B + action expert；重现论文模型数据训练需另行实现。
- 全量数据下载涉及速率限制和 3D 场景协议；评分脚本能打开不代表完整资产已正确安装。

## 对 wiki 的映射

- [主节点与源码运行时序图](../../wiki/entities/paper-behavior-spatial.md)
- [论文来源](../papers/behavior_spatial_arxiv_2610_34051.md)
- [项目页](../sites/behavior-spatial-benchmark.md)
- [数据集](../datasets/behavior-spatial-benchmark.md)
