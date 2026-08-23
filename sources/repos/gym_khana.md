# Gym-Khana

> 来源归档

- **标题：** Gym-Khana
- **类型：** repo
- **来源：** TeoIlie（GitHub）
- **链接：** https://github.com/TeoIlie/Gym-Khana
- **文档：** https://gym-khana.readthedocs.io
- **PyPI：** https://pypi.org/project/gymkhana/
- **Stars：** ~3（2026-08-23；活跃维护）
- **许可证：** MIT
- **入库日期：** 2026-08-23
- **一句话说明：** 基于 f1tenth_gym 的 Gymnasium 环境，用 SB3 + wandb 训练 1/10 或全尺寸 Ackermann 竞速与漂移；含课程学习、域随机化、ONNX 导出与 Sim2Real 文档。
- **代码：** https://github.com/TeoIlie/Gym-Khana（**已开源**）
- **沉淀到 wiki：** 景观页

---

## 核心定位

- **后端：** [f1tenth_gym](./f1tenth_gym.md) 单车动力学
- **RL：** Stable-Baselines3；可选 wandb
- **工程：** 自定义地图/轮胎参数、`gym.make()` 配置、控制/观测 debug 面板

---

## 典型入口

```bash
pip install gymkhana   # 或 pip install -e .
# 见 docs 与 examples
```

---

## 关联

- [`f1tenth_gym.md`](./f1tenth_gym.md)、[`xcar_rlgpu.md`](./xcar_rlgpu.md)
