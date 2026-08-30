# spot-rl-example

> 来源归档

- **标题：** Boston Dynamics spot-rl-example
- **类型：** repo
- **来源：** Boston Dynamics（原 The AI Institute 开发，支撑 Spot RL Researcher Kit）
- **链接：** https://github.com/boston-dynamics/spot-rl-example
- **入库日期：** 2026-08-30
- **一句话说明：** Spot RL Researcher Kit 的 **Jetson 真机部署参考实现**：加载 Isaac Lab 导出的 **ONNX 策略**，经 Spot Python SDK joint-level API 与 PS4 手柄速度指令闭环运行。
- **代码：** https://github.com/boston-dynamics/spot-rl-example（**已开源**）
- **沉淀到 wiki：** 是 → [`wiki/entities/nvidia-isaac-lab-spot-locomotion-sim2real.md`](../../wiki/entities/nvidia-isaac-lab-spot-locomotion-sim2real.md)、[`wiki/entities/paper-spot-rl-distributional-sim2real.md`](../../wiki/entities/paper-spot-rl-distributional-sim2real.md)

---

## 核心定位

- **训练侧：** 在 Isaac Lab 用 `Isaac-Velocity-Flat-Spot-v0` + RSL-rl 训练；`play.py` 导出 `.onnx` 与 env 配置
- **部署侧：** Orin 上 clone 本仓 → 安装 `external/spot_python_sdk` 预编译 wheel → `python spot_rl_demo.py`
- **依赖：** Spot 带 joint-level API 的 Python SDK、onnxruntime、pygame / pyPS4Controller、spatialmath-python

---

## 关联档案

- NVIDIA 教程博客：[`nvidia_isaac_lab_spot_locomotion_sim2real.md`](../blogs/nvidia_isaac_lab_spot_locomotion_sim2real.md)
- Isaac Lab：[`isaac_lab.md`](./isaac_lab.md)
- Spot RL 论文：[`spot_rl_distributional_sim2real_arxiv_2504_17857.md`](../papers/spot_rl_distributional_sim2real_arxiv_2504_17857.md)
