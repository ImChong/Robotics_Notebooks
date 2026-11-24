# FastRoverVLA 官方源码

- **类型：** repo
- **仓库：** https://github.com/ADEPT-ORG/FastRoverVLA
- **论文：** https://arxiv.org/abs/2611.26242
- **项目：** https://adept-org.github.io/FastRoverVLA/
- **许可证：** MIT（代码；第三方数据另有条款）
- **核查日期：** 2026-11-24
- **开放状态：** 执行核心、全精度配置、1286 窗口数值重放及控制实验已发布，原始 GPU 推理链未发布。
- **核对范围：** README、src/fastrovervla/execution.py、refresh.py、evaluation.py、scripts/reproduce.py。
- **通读：** 已完成

## 运行入口

```bash
git clone https://github.com/ADEPT-ORG/FastRoverVLA.git
cd FastRoverVLA
python -m pip install --no-deps -e .
python -m fastrovervla demo --steps 24
python scripts/reproduce.py --verify
```

- `scripts/reproduce.py`：读 JSONL 与配置，调用 replay，验证轨迹、刷新计划与指标；输出 summary.json、summary.csv、verification.json。
- `evaluation.py`：对每个记录构造 FrameContext，调度器及执行策略调用 step，计算 metrics；模型回调读取冻结动作。
- `execution.py`：FastRoverPolicy.step，序列切换重置；仅刷新调用 planner；保存修正尾部。
- `refresh.py`：RefreshConfig.score/budget，基于未补齐动作差分，验证集 P90 归一化。
- 官方预期 1286 窗口、37/31 次 adaptive admission，主要轨迹误差小于 1e-12；本次仅核对源码，未运行复现。

- [论文归档](../papers/fastrovervla_arxiv_2611_26242.md)
- [项目页归档](../sites/fastrovervla.md)
- [知识节点](../../wiki/entities/paper-fastrovervla.md)
