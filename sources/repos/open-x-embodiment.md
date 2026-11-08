# Open X-Embodiment 官方仓库

- **代码：** https://github.com/google-deepmind/open_x_embodiment
- **项目页：** https://robotics-transformer-x.github.io/
- **论文：** https://arxiv.org/abs/2310.08864v9
- **核对快照：** 9eeb68b989efbcf474e8fb9019e01d02b962a604
- **核对日期：** 2026-11-08
- **核对范围：** 阅读官方 README（Dataset Access、RT-1-X checkpoint、FAQ、License）和 `models/rt1_inference_example.py` 全文件；查看 `models/` 官方代码目录。未运行 GPU 推理或真实机器人部署。
- **通读：** 已完成

README 提供 RLDS 数据读取 Colab、TensorFlow 离线推理 Colab、RT-1-X JAX checkpoint 和 JAX 示例。软件许可为 Apache 2.0；其他资料依仓库声明及各源数据许可核对。

JAX [models/rt1_inference_example.py](https://github.com/google-deepmind/open_x_embodiment/blob/9eeb68b989efbcf474e8fb9019e01d02b962a604/models/rt1_inference_example.py) 的 main 构造 RT1 与 RT1Policy，从 checkpoint 恢复变量，生成假图像序列和语言嵌入，调用 action；后者缩放图像，JIT 调用模型 apply，再对 logits 取 argmax 并调用 detokenize_action。示例不实现真机控制。项目页见 [项目页归档](../sites/open-x-embodiment.md)，论文见 [论文归档](../papers/hmi_p055_open-x-embodiment.md)。

## 官方可执行入口及边界（源码核查）

- **数据读取：** README 的 `colabs/Open_X_Embodiment_Datasets.ipynb` 以 RLDS episode 格式展示读取、可视化和训练用 batch 的生成；数据集清单链接官方 Google Sheet。
- **RT-1-X 权重：** README 给出 `gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_jax .` 下载方式；本次未实际下载或运行权重。
- **官方 Python 入口：** [`models/rt1_inference_example.py`](https://github.com/google-deepmind/open_x_embodiment/blob/9eeb68b989efbcf474e8fb9019e01d02b962a604/models/rt1_inference_example.py)，`app.run(main)` 调用 `main` → 构造 `rt1.RT1` 和 `RT1Policy` → `flax.training.checkpoints.restore_checkpoint` → 创建 15 帧假图像和 15 帧预计算语言嵌入 → `policy.action(obs)`。
- **policy.action：** 复制观测、删除不支持的文本键（若存在）、`tf.image.resize` 为 300×300 并归一化、拆分 JAX RNG、JIT 调用 `_run_action_inference`；后者创建零动作 token、添加 batch 维度，`rt1.RT1.apply` 得 logits，softmax/argmax 后 `rt1.detokenize_action`；设备取回并修正不合法的终止标记后返回动作字典。
- **注意：** 示例明确是**合成观测的离线推理**。不能据此断言全量训练代码、RT-2-X 55B 权重或真机闭环驱动已公开。不同源数据的原始许可和引用条件仍需单独核验。
- 对照论文：[v9 一手归档](../papers/open-x-embodiment-arxiv-2310-08864-v9.md)。
