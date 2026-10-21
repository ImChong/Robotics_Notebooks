# RoboValue 官方代码仓库

> 来源归档（ingest）

- **类型：** 官方评测代码仓库
- **仓库：** <https://github.com/RoboValue-Benchmark/RoboValue>
- **默认分支：** `main`
- **论文：** <https://arxiv.org/abs/2610.26663>
- **项目页：** <https://robovalue-benchmark.github.io/>
- **许可：** 仓库自有代码、脚本和文档采用 MIT；第三方代码、模型权重与数据分别遵循各自条款
- **核查日期：** 2026-10-20
- **一句话说明：** 提供 RoboValue 的模型适配器接口、指标、数据工具和评测运行器；完整排行榜测试集仍由组织方托管。

## 仓库入口

- `src/vmbmk/adapters/base.py`：模型推理适配器接口。
- `src/vmbmk/adapters/mock_service.py`：CPU mock service，可用于本地合成数据接口检查。
- `src/vmbmk/inference/`、`src/vmbmk/runner/`：查询推理、批处理与评测运行逻辑。
- `src/vmbmk/metrics/`：查询规划与细粒度指标计算。
- `configs/`、`docs/`：评测配置及数据、环境和接入文档。

README 给出的公开 CPU 检查命令会生成临时数据并验证 mock adapter；它不调用参评模型服务，也不产出官方评测。官方测试集不公开下载，组织方负责准备查询、运行适配器并计算发布指标。

## 对 wiki 的映射

- [RoboValue 统一实体页](../../wiki/entities/paper-robovalue-benchmark.md)
- [项目主页归档](../sites/robovalue-benchmark.md)
