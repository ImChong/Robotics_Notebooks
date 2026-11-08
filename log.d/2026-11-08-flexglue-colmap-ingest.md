## [2026-11-08] ingest | FlexGlue 现有实体补充 COLMAP ONNX 官方集成

- 去重：沿用已合并 PR #3418 建立的 `paper-flexglue`，不新增项目节点。
- 核对一手资料：arXiv:2610.32976v1、cvg/FlexGlue 推理源码、COLMAP 官方 Features 文档、`src/colmap/feature/flexglue.{h,cc}`。
- 补充 COLMAP ONNX 描述器与匹配器的 C++ 源码运行时序图、构建条件、关键参数与复现边界。
- 修正 PyTorch 默认关闭与 COLMAP 默认开启共视过滤的差异；COLMAP 官方文档的 35% 早停实验仅适用于指定 ETH3D 配置。
- 原论文全文及定性图尚未完成逐段逐图核查；来源标为未完成，PR 为草稿；未在当前环境运行 `make ci-preflight` 或执行模型。
