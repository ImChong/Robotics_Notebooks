# TensorFlow 官方站点与文档索引

> 来源归档（以官网与 GitHub README 叙述为准；版本号与安装命令以安装页实时输出为准）

- **标题：** TensorFlow — An end-to-end platform for machine learning
- **类型：** 官方站点 + 文档 + 教程 + 开源核心仓库
- **主页：** https://www.tensorflow.org/
- **安装指南：** https://www.tensorflow.org/install
- **pip 安装：** https://www.tensorflow.org/install/pip
- **GPU 支持：** https://www.tensorflow.org/install/gpu
- **从源码构建：** https://www.tensorflow.org/install/source
- **教程：** https://www.tensorflow.org/tutorials/
- **API 文档（Python）：** https://www.tensorflow.org/api_docs/python
- **核心代码：** https://github.com/tensorflow/tensorflow
- **入库日期：** 2026-06-25
- **一句话说明：** 由 Google Brain 团队发起、现由社区维护的 **端到端开源机器学习平台**；以 **Keras 高层 API** 降低建模门槛，并通过 **TFX / LiteRT / TensorFlow.js** 等子项目覆盖 **生产 MLOps、移动与边端、浏览器** 等部署形态；在机器人栈中常见于 **嵌入式推理（TFLite/LiteRT）** 与部分 **感知/RL 遗留代码**。
- **沉淀到 wiki：** [TensorFlow](../../wiki/entities/tensorflow.md)

---

## 首页要点（2026-06-25 抓取归纳）

1. **定位**：「An end-to-end platform for machine learning」—— 从交互式教程、标准数据集到生产工具链的一体化叙事。
2. **入门路径**：首页以 MNIST + `tf.keras` Sequential 示例展示最小训练闭环（`model.fit` / `model.evaluate`）。
3. **应用案例**：突出 **TensorFlow.js**（Web 端推理）、**TensorFlow GNN**（关系数据/图神经网络）、**TensorFlow Agents**（强化学习与推荐系统，如 Spotify 播放列表案例）。
4. **生态组件**（首页 Ecosystem 列举）：
   - **TensorFlow.js**：浏览器或 Node.js 中训练与运行模型
   - **LiteRT**（原 TensorFlow Lite 品牌演进）：Android、iOS、Raspberry Pi、Edge TPU 等边端部署
   - **tf.data**：数据预处理与输入管线
   - **TFX**：生产级 ML 流水线与 MLOps
   - **tf.keras**：高层建模 API
   - **TensorFlow Datasets**、**Kaggle Models**、**TensorBoard** 等配套资源

---

## GitHub README 要点（tensorflow/tensorflow）

- **起源**：最初由 Google Brain 的 Machine Intelligence 团队为 ML 与神经网络研究而开发；框架本身可泛化到其他领域。
- **语言 API**：提供稳定的 **Python** 与 **C++** API；其他语言有 **不保证向后兼容** 的 API 层。
- **安装**：
  - GPU（CUDA，Ubuntu/Windows）：`pip install tensorflow`
  - 仅 CPU 小包：`pip install tensorflow-cpu`
  - 测试通道：`tf-nightly` / `tf-nightly-cpu`（PyPI）
  - 其他设备（DirectX、macOS Metal）通过 **Device Plugins** 支持
- **许可**：Apache License 2.0
- **社区**：GitHub Issues 跟踪 bug；一般讨论见 TensorFlow Forum；具体问题可上 Stack Overflow

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [深度学习基础](../../wiki/concepts/deep-learning-foundations.md) | 与 PyTorch 并列的主流教学与实现栈之一；Keras API 降低入门门槛 |
| [强化学习](../../wiki/methods/reinforcement-learning.md) | TensorFlow Agents 等子生态支持 RL；机器人 RL 研究侧 PyTorch 更常见，边端导出常经 TFLite |
| [htwk-gym](../../wiki/methods/htwk-gym.md) | Booster 人形足球栈用 **TFLite 量化** 将策略部署到 ARM/Orin |
| [PyTorch](../../wiki/entities/pytorch.md) | 训练侧主流对照框架；二者在 ONNX、TFLite 等格式上可衔接部署 |

---

## 对 wiki 的映射

- 新建 **`wiki/entities/tensorflow.md`**：框架实体页（能力边界、边端部署、与机器人训练/部署关系、互链）。
- 轻量更新 **`wiki/concepts/deep-learning-foundations.md`**：补充交叉引用，避免孤岛页。

---

## 外部参考（便于复核）

- [TensorFlow 官网](https://www.tensorflow.org/)
- [Install TensorFlow](https://www.tensorflow.org/install)
- [Tutorials](https://www.tensorflow.org/tutorials/)
- [API Docs (Python)](https://www.tensorflow.org/api_docs/python)
- [tensorflow/tensorflow（GitHub）](https://github.com/tensorflow/tensorflow)
- [LiteRT（边端）](https://www.tensorflow.org/lite)
- [TensorFlow.js](https://www.tensorflow.org/js)
