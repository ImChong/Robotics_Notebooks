# AccumulateMore/CV — 深度学习、计算机视觉、NLP 与 Agent Jupyter 笔记

- **类型：** repo / 中文课程学习笔记（Jupyter Notebook 资料集）
- **官方仓库：** https://github.com/AccumulateMore/CV
- **维护者：** AccumulateMore
- **核对版本：** [main@e05116c8336e8dd4da56118970ae45a3e358e0d3](https://github.com/AccumulateMore/CV/tree/e05116c8336e8dd4da56118970ae45a3e358e0d3)（提交时间 2026-09-20，核对日期 2026-11-22）
- **核对范围：** 通读 README；核查根目录全部文件名与课程编号；逐单元阅读 `103_Pytorch加载数据.ipynb`、`116_优化器.ipynb`、`214_PyTorch神经网络基础.ipynb`、`402_向量数据库.ipynb`、`403_基于向量的语义检索RAG.ipynb`、`404_代码分析与展望.ipynb`、`405_使用LangChain构建RAG.ipynb`、`408_API基本参数.ipynb`、`409_多轮对话.ipynb`。未逐页核查仓库其他大型 Notebook 的代码、图片与输出。
- **通读：** 已完成
- **开源状态：** 原始 Notebook 文本与部分代码单元公开可读；属于课程笔记合集，不是提供统一入口、打包依赖与自动化测试的训练框架。
- **许可证：** 根目录文件清单未见 `LICENSE`；未确认独立再分发许可，优先引用链接，不批量复制教程与图片。

## 原文定位

上游 README 的标题为「超级全面的 深度学习 笔记（包含CV、NLP、大模型、Agent）」，把不同授课系列按编号串在同一仓库。仓库名 `CV` 不等于内容只涉及计算机视觉；同时包含深度学习数学、PyTorch、视觉网络、序列模型和大模型应用接口。

| 编号 | README 给出的主讲与课程索引 | 可从文件名核对的学习主题 |
| --- | --- | --- |
| 100–122 | 土堆 / PyTorch | Dataset、DataLoader、TensorBoard、Transforms、CNN、训练与验证、GPU |
| 200–268 | 李沐 / 深度学习 | 线性代数、自动求导、MLP、CNN/ResNet、检测分割、RNN/Transformer/BERT、优化和分布式 |
| 300–354 | 吴恩达 / 深度学习 | 神经网络、正则与优化、机器学习策略、CNN、序列模型及配套习题 |
| 400–409 | 大飞 / 大模型 Agent | 向量数据库、RAG、LangChain、对话 API、系统提示、温度与多轮对话 |
| 500–5XX | README 中的未来规划 | 原文仍为占位，不视为已交付课程 |

## 逐单元核验的代表性实现

1. [103 数据加载](https://github.com/AccumulateMore/CV/blob/e05116c8336e8dd4da56118970ae45a3e358e0d3/103_Pytorch加载数据.ipynb)：通过 `torch.utils.data.Dataset` 的 `__getitem__` / `__len__`，按目录读取 PIL 图片与文件夹标签；示例使用相对路径 `Data/FirstTypeData/train`，运行前要准备数据。
2. [116 优化器](https://github.com/AccumulateMore/CV/blob/e05116c8336e8dd4da56118970ae45a3e358e0d3/116_优化器.ipynb)：`torchvision.datasets.CIFAR10` → `DataLoader(batch_size=64, drop_last=True)` → `Tudui(nn.Module)` 的 3 组 Conv/Pooling + Linear 分类头 → `CrossEntropyLoss` → `SGD.zero_grad()`、`loss.backward()`、`SGD.step()`。多轮示例循环 20 个 epoch。
3. [214 PyTorch 网络模块](https://github.com/AccumulateMore/CV/blob/e05116c8336e8dd4da56118970ae45a3e358e0d3/214_PyTorch神经网络基础.ipynb)：`nn.Sequential`、自定义 `nn.Module.forward`、嵌套网络与参数初始化。
4. [402 向量表示](https://github.com/AccumulateMore/CV/blob/e05116c8336e8dd4da56118970ae45a3e358e0d3/402_向量数据库.ipynb)：用 `SentenceTransformer('all-MiniLM-L6-v2')` 编码句子并调用 `model.similarity`；标题虽称向量数据库，本例本身主要演示嵌入与相似度，并无数据库持久化。
5. [403 RAG 语义检索](https://github.com/AccumulateMore/CV/blob/e05116c8336e8dd4da56118970ae45a3e358e0d3/403_基于向量的语义检索RAG.ipynb)：通过 ModelScope 获取 `BAAI/bge-large-zh-v1.5`，再以 `SentenceTransformer` 编码本地文本并使用 `chromadb.PersistentClient` 建集、查询。
6. [404/405 RAG 应用](https://github.com/AccumulateMore/CV/blob/e05116c8336e8dd4da56118970ae45a3e358e0d3/404_代码分析与展望.ipynb)：组合 BGE/Chroma 检索、提示词与兼容 OpenAI 协议的 DeepSeek Chat API；[405](https://github.com/AccumulateMore/CV/blob/e05116c8336e8dd4da56118970ae45a3e358e0d3/405_使用LangChain构建RAG.ipynb) 进一步用 LangChain 组件建立检索链。
7. [408/409 Agent API](https://github.com/AccumulateMore/CV/blob/e05116c8336e8dd4da56118970ae45a3e358e0d3/408_API基本参数.ipynb)：系统消息、流式输出、temperature；[409](https://github.com/AccumulateMore/CV/blob/e05116c8336e8dd4da56118970ae45a3e358e0d3/409_多轮对话.ipynb) 通过维护 `messages` 列表实现两轮对话上下文。

## 实践边界与已核对风险

- **数据划分：** 116 的三个训练示例都从 `CIFAR10(..., train=False, ...)` 获取数据并更新模型，不能当成符合训练/测试隔离的标准评测范例；实际练习需改用训练集训练、独立测试集报告指标。
- **调度步长：** 116 的 `StepLR(step_size=5, gamma=0.1)` 示例把 `scheduler.step()` 放在每个 batch 的内部循环里；这意味着每 5 个 batch 而不是每 5 个 epoch 降一次学习率，应按实际训练意图调整。
- **依赖与数据：** 多个 Notebook 依赖本地 `Data/`、`model_dir/`、`my_knowledge/`、`DEEPSEEK_API_KEY`、外部数据集与网络下载；没有针对全仓统一验证的锁定环境。
- **工程范围：** 可作为 PyTorch/感知骨干基础练习，不能推定已经提供机器人策略、Sim2Real 实验或生产可用 RAG 系统；当前仅对上面列出的代表性 Notebook 做源码逐单元核查。
- **许可：** 笔记涉及多个课程来源与演示内容；原始仓库根目录未见 LICENSE，引用和复用应核对原作者、授课材料及第三方数据各自许可。

## 对 Wiki 的映射

- [AccumulateMore/CV 唯一项目实体](../../wiki/entities/accumulatemore-cv.md) — 学习路径、流程、代表性运行时序图。
- [深度学习基础](../../wiki/concepts/deep-learning-foundations.md) — 对应数学、网络与优化概念。
- [模型架构纵深路线](../../roadmap/depth-model-architecture.md) — Stage 0 的 PyTorch 先修和 Stage 1 的骨干知识。

## 一手资料

- [官方 README](https://github.com/AccumulateMore/CV/blob/e05116c8336e8dd4da56118970ae45a3e358e0d3/README.md)
- [源仓库固定版本](https://github.com/AccumulateMore/CV/tree/e05116c8336e8dd4da56118970ae45a3e358e0d3)
- [PyTorch 优化器 Notebook](https://github.com/AccumulateMore/CV/blob/e05116c8336e8dd4da56118970ae45a3e358e0d3/116_优化器.ipynb)
- [RAG 语义检索 Notebook](https://github.com/AccumulateMore/CV/blob/e05116c8336e8dd4da56118970ae45a3e358e0d3/403_基于向量的语义检索RAG.ipynb)
