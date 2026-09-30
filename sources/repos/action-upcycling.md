# star-kwon/action-upcycling

> 来源归档（repo）

- **名称：** action-upcycling
- **类型：** repo / vla / deployment / action-chunking / openpi
- **URL：** <https://github.com/star-kwon/action-upcycling>
- **论文：** [arXiv:2609.34911](../papers/action_upcycling_arxiv_2609_34911.md)
- **项目页：** <https://acupcycling.github.io/> — [`sources/sites/acupcycling-github-io.md`](../sites/acupcycling-github-io.md)
- **机构：** Sungkyunkwan University、KAIST
- **许可证：** Apache-2.0（基于 OpenPI 端口）
- **入库日期：** 2026-09-30
- **一句话说明：** Action Upcycling 官方实现：OpenPI π0.5 LIBERO policy server + `run_upcycling.sh` baseline/upcycling 对照与 `summarize.py` 汇总。

## 运行入口（README）

| 步骤 | 脚本 / 模块 |
|------|-------------|
| Policy server | `uv run scripts/serve_policy.py --env LIBERO` |
| Baseline 评测 | `bash examples/libero/run_upcycling.sh baseline` |
| Upcycling 评测 | `bash examples/libero/run_upcycling.sh upcycling --config pi05_libero_r1.5` |
| 结果对比 | `python examples/libero/summarize.py results/baseline results/upcycling` |

## 关键路径

| 路径 | 作用 |
|------|------|
| `examples/libero/run_upcycling.sh` | baseline vs upcycling 协议切换 |
| `examples/libero/summarize.py` | 成功率与 calls/ep 汇总 |
| `src/openpi/` | OpenPI 策略与 chunk 执行逻辑（含 upcycling 门控） |
| `scripts/serve_policy.py` | LIBERO websocket 推理服务 |

## 开源边界（2026-09-30）

- **已开源：** LIBERO 四套件评测管线、upcycling 配置（如 `pi05_libero_r1.5`）
- **外部依赖：** OpenPI / π0.5 checkpoint（`gs://openpi-assets/...` 或用户自备）
- **未在仓内：** RoboTwin / 真机 YAM 脚本以论文为准；仿真主表可经 LIBERO 复现核心 claim
