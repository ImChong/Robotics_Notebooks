# Open X-Embodiment 官方仓库

- **代码：** https://github.com/google-deepmind/open_x_embodiment
- **项目页：** https://robotics-transformer-x.github.io/
- **论文：** https://arxiv.org/abs/2310.08864v9
- **核对快照：** 9eeb68b989efbcf474e8fb9019e01d02b962a604
- **核对日期：** 2026-11-05

README 提供 RLDS 数据读取 Colab、TensorFlow 离线推理 Colab、RT-1-X JAX checkpoint 和 JAX 示例。软件许可为 Apache 2.0；其他资料依仓库声明及各源数据许可核对。

JAX [models/rt1_inference_example.py](https://github.com/google-deepmind/open_x_embodiment/blob/9eeb68b989efbcf474e8fb9019e01d02b962a604/models/rt1_inference_example.py) 的 main 构造 RT1 与 RT1Policy，从 checkpoint 恢复变量，生成假图像序列和语言嵌入，调用 action；后者缩放图像，JIT 调用模型 apply，再对 logits 取 argmax 并调用 detokenize_action。示例不实现真机控制。项目页见 [项目页归档](../sites/open-x-embodiment.md)，论文见 [论文归档](../papers/hmi_p055_open-x-embodiment.md)。
