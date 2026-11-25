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
- **核对范围：** 官方 README、pyproject.toml、docs/METHOD.md、DEPLOYMENT.md、VALIDATION.md、MODEL_RELEASE.md；通读公开推理主干 cli.py、replay.py、policy.py、runtime.py、schedule.py、config.py、VASE/head.py、VASE/runtime.py、TACE/runtime.py、TACE/score.py、TACE/refiner.py、forecast/service.py、forecast/verifier.py、forecast/execution.py、forecast/model.py、forecast/shared_encoder.py 与 src/openpi/fastslow/model.py。重点为论文核心方法、回放与执行保护路径，未复现训练或运行权重，未把作者验证当成本站实测。
- **通读：** 已完成

## 运行入口与调用

- pyproject.toml 注册 tipdex = tipdex.cli:main；CLI verify 使用 ModelBundle.load，replay 创建 TipDexPolicy 后进入 run_replay，serve 进入 serve_policy。
- run_replay 载入样例，reset，feed_tactile，slow；forecast.predict 后构造模拟 SDK ACK、送入触觉，fast；写 report.json 和 prediction.npz。
- TipDexPolicy.slow 更新 VisualReliability 并缓存 forecast context；fast 调用 forecast.verifier.score，设置 TaceAdapter 的 q 与 slot_mask，随后调用基类快端。
- ClosedLoop.slow_update、fast_update、issue_forecast 与 commit 管理执行时间和可替换位置。真实 SDK ACK 由外部驱动提供，不能从候选动作伪造。
- 官方 Linux/Python 3.11/CUDA 12，A100 80GB 验证；tipdex replay --model MODEL_DIR --config paper --output outputs/paper。回放不是实机精度评测。
- 30×13 动作、19 维状态、三路 RGB、最近 16 帧 raw364；14×26 热图是寄存器顺序，不是已标定力图。

## 核心实现补核

- [METHOD.md](https://github.com/tiplab/TipDex-VTLA/blob/main/docs/METHOD.md) 给出论文定义到源码的映射。VASE 的 masked mean 视觉前缀经 2048→128→1 LayerNorm/MLP 输出 sigmoid；按 session/chunk 缓存。六个慢流步后缓存反向流时间 t≈0.4，四个快流步从同一快照续流，避免递归使用前次修订。
- [forecast/model.py](https://github.com/tiplab/TipDex-VTLA/blob/main/src/tipdex/forecast/model.py) 以共享触觉编码器生成 16×256 token；GRU 历史端融合状态、命令和时间，未来端条件化未来命令，输出 H×364。共享编码器使用同一对象，按参数摘要检查一致性。
- [forecast/verifier.py](https://github.com/tiplab/TipDex-VTLA/blob/main/src/tipdex/forecast/verifier.py) 仅挑最新合格当前帧预测，检查采集/接收时钟、签发时间、目标时刻和命令历史；[execution.py](https://github.com/tiplab/TipDex-VTLA/blob/main/src/tipdex/forecast/execution.py) 检查所有中间 SDK ACK 及计划转换，缺失或重启使证据失效，非测量位姿。
- [TACE/score.py](https://github.com/tiplab/TipDex-VTLA/blob/main/src/tipdex/TACE/score.py) 对已核验误差作平滑阈值映射，扣分上限为配置最大扣分与基准分一半中的较小者；无有效预测回退视觉基准。paper 与 tipdex 预设分别采用 H=30/5、残差上界 0.1/0.01、视觉映射 qv / 0.6+0.4*qv。
- [TACE/refiner.py](https://github.com/tiplab/TipDex-VTLA/blob/main/src/tipdex/TACE/refiner.py) 在最后 suffix Transformer 输出、动作投影前作共享 64 维 GELU trunk 的特征修正；FP32 双积分流逐步恢复不可替换位置，避免 attention 跨位置传播；外部调度再次保护已提交动作。
- [VALIDATION.md](https://github.com/tiplab/TipDex-VTLA/blob/main/docs/VALIDATION.md) 是作者发布验收：49 项自动测试；两种配置均完成两次接受更新/重新预测，保护位置变化为零。模拟 ACK 和注入误差只验证执行路径，非真机成功率或预测精度。

## 对 wiki 的映射

- [TipDex-VTLA](../../wiki/entities/paper-tipdex-vtla.md)
- [论文归档](../papers/tipdex_vtla_arxiv_2611_25683.md)
- [项目页归档](../sites/tipdex-vtla.md)
