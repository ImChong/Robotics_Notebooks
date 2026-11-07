# ONNX 官方站点与规范索引

> 来源归档（以 onnx.ai 官网、GitHub README 与 1.22.x 文档叙述为准；算子集与版本以目标 runtime 文档为准）

- **标题：** ONNX — Open Neural Network Exchange
- **类型：** 开放标准 + 规范文档 + Python 参考实现
- **主页：** https://onnx.ai/
- **文档：** https://onnx.ai/onnx/
- **规范仓库：** https://github.com/onnx/onnx
- **预训练模型集：** https://github.com/onnx/models
- **教程仓库：** https://github.com/onnx/tutorials
- **支持的工具与框架列表：** https://onnx.ai/supported-tools
- **治理：** LF AI & Data Foundation **Graduate Project**（开放治理、SIG / Working Group）
- **入库日期：** 2026-06-25；**续更：** 2026-07-26（对齐 ORT 1.28.0 捆绑 ONNX 1.22.0）
- **复核日期：** 2026-11-07（重新取得 raw.githubusercontent.com 上 onnx/onnx main 分支文档与 PyPI 元数据）
- **核对范围：** `README.md`、`docs/Overview.md`、`docs/IR.md`（Components、Model/Graph/Node/Function/Attribute 各结构、标准数据类型、静态形状、外部数据、训练信息、元数据）、`docs/Versioning.md`（版本原则、opset 绑定）、`docs/PythonAPIOverview.md`（加载/外部数据/保存）读完；`docs/Operators.md`（1.3 MB 自动生成的算子大全）只读表头与抽查若干算子的 since_version，不逐个算子通读；`onnx.proto` 与 checker 源码未读；`docs/OnnxReleases.md` 为发布流程，只读开头。
- **通读：** 已完成
- **一句话说明：** **开放神经网络交换格式（IR）**：用统一的计算图、算子集与 `.onnx` 文件格式，让开发者在 **PyTorch / TensorFlow / JAX** 等框架中训练后，把模型交给 **ONNX Runtime、TensorRT、MNN** 等推理引擎与硬件优化栈；当前规范侧重 **推理（scoring）** 能力，并含 **ONNX-ML** 扩展以覆盖经典 ML 算子。[ONNX Runtime 1.28.0](./onnxruntime-v1.28.0.md) 捆绑 **ONNX 1.22.0**。
- **沉淀到 wiki：** [ONNX](../../wiki/entities/onnx.md)

---

## 首页要点（2026-06-25 抓取归纳）

1. **定位**：「The open standard for machine learning interoperability」—— 定义通用 **operators** 与 **file format**，衔接多种框架、工具、runtime 与编译器。
2. **核心价值**：
   - **Interoperability**：在偏好框架中开发，不绑定下游推理栈。
   - **Hardware Access**：经 ONNX 兼容 runtime 触达各硬件优化路径。
3. **社区**：Slack、SIG、Working Group、贡献指南；LF AI 毕业项目身份。

---

## GitHub / 文档要点（onnx/onnx）

- **使命**：提供开源 **AI 模型格式**（深度学习 + 传统 ML）；定义可扩展 **computation graph**、内置 **operators** 与标准 **data types**。
- **变体**：
  - **ONNX**：神经网络推理所需核心能力。
  - **ONNX-ML**：额外类型与算子，服务经典机器学习算法标准化。
- **规范文档**（仓库 `docs/` / 官网）：
  - [Overview](https://github.com/onnx/onnx/blob/main/docs/Overview.md)
  - [IR Specification](https://github.com/onnx/onnx/blob/main/docs/IR.md)
  - [Operators](https://onnx.ai/onnx/operators/index.html)
  - [Versioning](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)
  - [Python API Overview](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)
- **安装**：`pip install onnx`（可选 `onnx[reference]` 含参考实现依赖）；Apache License 2.0。
- **生态**：广泛被框架、工具与硬件支持；降低「研究框架 → 生产 runtime」摩擦。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [PyTorch](../../wiki/entities/pytorch.md) | `torch.onnx.export` 等路径将训练图导出为 `.onnx`；机器人 RL 策略上机常见中间格式 |
| [TensorFlow](../../wiki/entities/tensorflow.md) | `tf2onnx` 等工具链可转入 ONNX，再交给 ORT / TensorRT |
| [ONNX Runtime](../../wiki/entities/onnxruntime.md) | 微软主导的 **生产级 ONNX 推理引擎**；与 ONNX **格式规范** 分工不同 |
| [MNN](../../wiki/entities/mnn.md) | 阿里边端引擎；`mnnconvert -f ONNX` 将 ONNX 转为 `.mnn` |
| [Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md) | 人形 tracking 真机层普遍写「ONNX / TensorRT @ 50 Hz」 |
| [Sim2Real](../../wiki/concepts/sim2real.md) | 训练框架 ≠ onboard runtime；ONNX 是 sim2real **部署契约** 之一 |

---

## 对 wiki 的映射

- 新建 **`wiki/entities/onnx.md`**：格式/IR 实体页（与 runtime 区分、机器人导出注意点）。
- 新建 **`wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md`**：边端/机载推理 runtime 选型对比。
- 轻量交叉更新 **`wiki/entities/pytorch.md`**、**`wiki/concepts/sim2real.md`**、**`wiki/concepts/whole-body-tracking-pipeline.md`**。

---

## 外部参考（便于复核）

- [ONNX 官网](https://onnx.ai/)
- [ONNX 1.22 文档](https://onnx.ai/onnx/)
- [onnx/onnx（GitHub）](https://github.com/onnx/onnx)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [Supported Tools](https://onnx.ai/supported-tools)

## 2026-11-07 全文复核补充（onnx/onnx main + PyPI）

- **版本：** PyPI 最新 `onnx` 为 **1.23.2**（2026-10-06）；main 分支 `VERSION_NUMBER` 为 1.24.0；README 称 wheel 为 abi3、Python 3.12 起。onnxruntime `main` 的 `cmake/deps.txt` 已升到 onnx v1.23.2（v1.28.0 到 v1.31.1 均为 onnx v1.22.0，见 [onnxruntime-official.md](./onnxruntime-official.md)）。
- **定位（README）：** 开放生态，为 deep learning 与传统 ML 提供开源模型格式；定义可扩展计算图、内置算子与标准数据类型；**当前重点是推理（scoring）能力**。
- **IR 规范（IR.md）：** 三个组件——可扩展计算图模型、标准数据类型、内置算子；前两者合称 IR。**IR 版本 6 及以前只覆盖推理；自 IR 版本 7 起模型格式也支持训练**（TrainingInfoProto）；推理模型的语义是**无状态函数**（除随机数状态）。`ModelProto` 含 `ir_version`、`opset_import`、`producer_name/version`、`graph`、`metadata_props`、`functions`，IR ≥11 增加多设备 `configuration`。**实现必须支持模型 opset_import 中的全部算子，否则拒绝该模型。**
- **图：** 无副作用计算；节点拓扑有序、无环、输出名满足 SSA；`initializer` 与同名 graph input 表示默认值；顶层图输入输出**必须有形状（至少给出秩）**，维度可为整数 `dim_value` 或符号 `dim_param`（动态轴）；大常量张量可存**外部数据**文件（路径相对模型文件）。
- **元素类型：** float16/32/64、bfloat16、多种 float8、float4/6、int2–int64、uint2–uint64、complex、string、bool。
- **版本规则（Versioning.md）：** IR 版本、opset、模型分别版本化；算子以 (domain, op_type, since_version) 标识；已发布为 STABLE 的算子语义不得改变；不同 opset 的算子集合是不可变快照；用户在 `opset_import` 声明所需 (domain, version)，空域名 `""` 即 ONNX 标准域。
- **工具（PythonAPIOverview）：** `onnx.load` / `onnx.save`、`load_external_data_for_model`、`convert_model_to_external_data`；`onnx.checker` 做通用校验（IR.md「Notes on model validation」）。
- **Operators.md：** main 分支已出现 opset 28 的条目（如 BitShift-28、Cast-28、Compress-28）。
