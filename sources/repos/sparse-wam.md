# Sparse-WAM 官方推理实现 | Official inference implementation

- **类型：** repo
- **代码：** https://github.com/Rockylix/Sparse-WAM
- **固定提交：** [eef61f5](https://github.com/Rockylix/Sparse-WAM/tree/eef61f5afadcfd159ff80350c442a4f17bc1c940)
- **项目页：** https://rockylix.github.io/Sparse-WAM.github.io/
- **论文：** https://arxiv.org/abs/2609.38984v1
- **许可：** OpenMDW-1.1；权重、样本和依赖另看各自条款。
- **核对范围：** README、RTX 4090 指南、第三方许可说明、Edge CLI、原生 service 装配、runner、controller、速度缓存与 UniPC。
- **通读：** 已完成
- **核对日期：** 2026-11-25
- **对应节点：** [Sparse-WAM](../../wiki/entities/paper-sparse-wam.md)

| 阶段 | 固定实现入口 | 行为 |
|---|---|---|
| 准备 | `tools/prepare_smoke.py` | 准备作者指南指向的外部 Edge DROID checkpoint、Wan VAE 和录制样本，保留版本与哈希 |
| 运行 | `examples/infer_edge.py` → `create_service` | 从 NPZ 读取观测；用 native 服务预处理并装配单卡 runner |
| 稀疏化 | `runtime.py` → `controller.py` / `velocity.py` | 每个 chunk 新建控制器，在原生四步 UniPC 内复用选择和省略区域的预测 |
| 校验 | `examples/infer_edge.py`；`tools/benchmark_policy.py` | 输出有限值动作与 JSON；比较相同观测与种子的 dense/sparse 路径 |

README 首发支持 Cosmos3 Edge / RTX 4090。源码包含 Nano 适配和独立工具，但 Nano 和 FastWAM-Joint 未列为已验证支持。不是 NVIDIA 官方仓；不打包权重，论文真机微调权重和采集数据未核实发布。这里只核对实现，未跑 GPU。

关联：[项目页归档](../sites/sparse-wam.md)、[论文归档](../papers/sparse_wam_arxiv_2609_38984.md)。
