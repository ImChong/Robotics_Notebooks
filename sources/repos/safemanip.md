# SafeManip 官方监控实现 / Official Temporal Safety Monitor

- **仓库：** [chengyuehuang511/SafeManip](https://github.com/chengyuehuang511/SafeManip)
- **核验快照：** `cb9ca772c5a2e7fce9a0d44f966aabd3e2f5eb3b`
- **原始论文：** [arXiv:2605.12386v3](https://arxiv.org/abs/2605.12386v3) · [原文归档](../papers/safemanip_arxiv_2605_12386.md)
- **开放范围：** 监控器、谓词与规格、示例 monitor JSON、部分评测启动器及分析代码可读；完整仿真需依赖、评测模型和第三方资产，仓库明确不跟踪结果和权重。未运行 GPU 仿真。
- **复核入口：** README 的 Manual Monitor Invocation 指向 `SafeManip/monitor/run_monitor_on_privileged.py`；`main()` 解析轨迹和输出参数，调用 `monitor_rollout()`；该函数装载并规范化轨迹、初始化 `SymbolicMonitor` 与重复义务监控器，产出违例绑定、首次不接受帧和解释等 JSON 字段。
- **对 wiki 的映射：** [SafeManip](../../wiki/entities/paper-safemanip.md)
