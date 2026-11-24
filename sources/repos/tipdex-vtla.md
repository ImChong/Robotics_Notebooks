# tiplab/TipDex-VTLA

> 官方源码归档

- **类型：** repo
- **链接：** https://github.com/tiplab/TipDex-VTLA
- **论文：** https://arxiv.org/abs/2611.25683v2
- **项目页：** https://tiplab.github.io/TipDex-VTLA/
- **权重：** https://huggingface.co/TipDex-VTLA/TipDex-VTLA
- **数据集：** https://huggingface.co/datasets/TipDex-VTLA/TipDex-Data
- **许可证：** Apache-2.0；模型另受 Gemma 条款约束
- **核查日期：** 2026-11-24
- **核对范围：** README、pyproject.toml、src/tipdex/cli.py、replay.py、policy.py、runtime.py 相关入口与部署协议。未实际下载权重运行。
- **通读：** 未完成

## 运行入口与调用

- pyproject.toml 注册 tipdex = tipdex.cli:main；CLI verify 使用 ModelBundle.load，replay 创建 TipDexPolicy 后进入 run_replay，serve 进入 serve_policy。
- run_replay 载入样例，reset，feed_tactile，slow；forecast.predict 后构造模拟 SDK ACK、送入触觉，fast；写 report.json 和 prediction.npz。
- TipDexPolicy.slow 更新 VisualReliability 并缓存 forecast context；fast 调用 forecast.verifier.score，设置 TaceAdapter 的 q 与 slot_mask，随后调用基类快端。
- ClosedLoop.slow_update、fast_update、issue_forecast 与 commit 管理执行时间和可替换位置。真实 SDK ACK 由外部驱动提供，不能从候选动作伪造。
- 官方 Linux/Python 3.11/CUDA 12，A100 80GB 验证；tipdex replay --model MODEL_DIR --config paper --output outputs/paper。回放不是实机精度评测。
- 30×13 动作、19 维状态、三路 RGB、最近 16 帧 raw364；14×26 热图是寄存器顺序，不是已标定力图。

## 对 wiki 的映射

- [TipDex-VTLA](../../wiki/entities/paper-tipdex-vtla.md)
- [论文归档](../papers/tipdex_vtla_arxiv_2611_25683.md)
- [项目页归档](../sites/tipdex-vtla.md)
