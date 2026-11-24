# MemT-1.0 官方实现仓库

- **URL：** <https://github.com/MemTensor/MemT-1.0>
- **组织：** MemTensor
- **关联项目总览：** [MemT-1.0 项目页归档](../sites/memt-1-0.md)
- **关联论文：** [MemT-1.0 技术报告](../papers/memt_1_0_arxiv_2610_26825.md)
- **模型发行：** [Hugging Face collection](https://huggingface.co/collections/MemTensor/memt-1) · [ModelScope collection](https://www.modelscope.cn/collections/MemTensor/MemT-10)
- **Pro 权重：** [Hugging Face Pro](https://huggingface.co/MemTensor/MemT-1.0-Pro) · [ModelScope Pro](https://www.modelscope.cn/models/MemTensor/MemT-1.0-Pro)
- **更新日期：** 2026-11-24
- **入库日期：** 2026-10-20

## 一句话说明

官方仓库提供 MemT-1.0 推理代码和 SGLang 集成，模型族包含 Nano、Flash 与 Pro；仓库 README 提供 Nano 服务端与 Python 客户端的基本启动流程。

## 开放状态（2026-11-24）

- **代码：已公开。** 官方仓库含 inference 目录、服务启动脚本及客户端；README 示例使用 SGLang 服务端和 `memt_sglang` 客户端。
- **软件许可：PolyForm Noncommercial License 1.0.0。** 论文材料另用 CC BY-NC-SA 4.0；二者不能互相替代。第三方组件继续适用各自许可。
- **权重许可：分开核对。** Hugging Face 的 MemT-1.0-Flash model card 标注 MIT；官方仓库 README 明确说明模型权重、数据集及第三方材料不自动包含在仓库软件许可中。Pro 的 Hugging Face 卡片标注 openmdw-1.1；其他发行文件仍应检查各自条款。ModelScope 入口由官方 README 确认，页面本次未能直接读取。
- **复现边界：** README 的 Nano 示例要求 NVIDIA CUDA 与 Python 3.11 或更新版本，示例服务使用两张 GPU；完整训练需要论文所述专用基础设施，不能由推理示例推断为可单机复现。

## 模型规格（官方 README）

| 模型 | Backbone | 总参数 | 激活参数 | 可训练记忆参数 | 动态记忆状态 |
|------|----------|--------|----------|----------------|--------------|
| MemT-1.0-Nano | Qwen3.6-35B-A3B | 36B | 3B | 451M | 10M |
| MemT-1.0-Flash | DeepSeek-V4-Flash | 288B | 17B | 3.61B | 336M |
| MemT-1.0-Pro | NVIDIA-Nemotron-3-Ultra-550B-A55B | 554B | 59B | 4.23B | 13M |

## 快速入口

- 安装推理依赖：进入 `inference/`，按 README 安装 `.[server]`；独立客户端可安装 `.[client]`。
- Nano 示例：`sh start_memt_nano.sh /models/MemT-1.0-Nano`。
- Pro：`sh start_memt_pro.sh /models/MemT-1.0-Pro`，调用 `python -m memt_sglang.launch`，官方示例 8 GPU、TP8/EP8/DP1、BF16、上下文 16384、快照目录 `./memt-state`。这不是任意 8 GPU 都能装载 554B 参数的保证；未实机验证。
- Flash 设置：参阅 [官方 inference guide](https://github.com/MemTensor/MemT-1.0/tree/main/inference)。
- 权重入口见上述 Hugging Face 与 ModelScope collections；具体可用模型及条款以发行页为准。

## 源码核对入口

- [Pro 启动脚本](https://github.com/MemTensor/MemT-1.0/blob/4b08b5d54072b80221f22c912a0d560e02847295/inference/start_memt_pro.sh)
- [MemTSession](https://github.com/MemTensor/MemT-1.0/blob/4b08b5d54072b80221f22c912a0d560e02847295/inference/memt_sglang/client/session.py)：`chat` 取得 lease，`_generate` 组织 prompt 并调用 `client.query`，`_commit_turns` 重放 token 并提交；`save_state` 在事务内提交未保存历史、导出候选状态，发布成功后清除本地轮次。
- [官方 inference guide](https://github.com/MemTensor/MemT-1.0/blob/4b08b5d54072b80221f22c912a0d560e02847295/inference/README.md) 定义 `/memt/control` 与 `/generate` 的请求顺序以及恢复流程。

## 交叉链接

- [MemT-1.0 实体页](../../wiki/entities/paper-memt-1-0-memory-native-evolution.md)
- [项目总览归档](../sites/memt-1-0.md)
- [技术报告摘录](../papers/memt_1_0_arxiv_2610_26825.md)
