# Cartwheel Control：官方开源实现与运行链核验

> 一手代码归档（2026-11-12）

- **标题：** Cartwheel Control
- **类型：** repo
- **代码：** <https://github.com/Cartwhl/cartwheel-control>
- **权重：** <https://huggingface.co/andrew-cartwheel/cartwheel-control>
- **机构：** Cartwheel
- **核对范围：** 通读 README、pyproject.toml、cartwheel_control/play.py、policy.py、sim.py、motion.py、robots.py、viewer.py；未逐行审阅 viewer.html / quat.py，未执行 GPU 物理仿真。
- **通读：** 已完成
- **入库日期：** 2026-11-12
- **一句话说明：** 官方公开可运行的 50 Hz 跟踪推理、200 Hz MuJoCo Warp 仿真、评测和回放代码。

## 源码定位

| 入口或模块 | 代码中实际作用 |
|---|---|
| `pyproject.toml` | CLI `cartwheel-play = cartwheel_control.play:main`，Python >=3.12 |
| `play.py::main` | 下载机器人权重/模型/参考运动、配置仿真、运行并导出 |
| `play.py::rollout` | 每个 clip 一个模拟机器人，首帧 reset，然后逐周期观察和执行 |
| `policy.py::Policy` | 目标编码器 MLP 输出 32D latent、与观测历史融合后逐关节输出 |
| `policy.py::Inputs` | `proprio` / `goal` / `fell`：本体观测、未来参考动作、跌倒判定 |
| `motion.py::Motions` | 读取 50 Hz NPZ、计算根和关节速度、管理片段索引 |
| `sim.py::Sim` | 0.005 s 物理步、decimation 4、PD 位置执行，MuJoCo Warp/mjlab |
| `robots.py` | RP1 / Asimov-1 / G1 各自的 MJCF、默认姿态、PD 增益和动作 scale |
| `viewer.py::write` | 输出网页可视化数据 `poses.bin` / `manifest.json` / `robot.glb` / `index.html` |

## 源码核查

- `fetch()` 为官方 Hugging Face 资产使用固定快照 `24fc279af8f70097f503767b81db5384d779aae5`。
- 50 Hz 策略周期里 `Sim.step` 执行 4 次 5 ms 物理更新。输出的是关节位置偏置，经机器人 scale 和默认姿态转换成 PD 位置目标。
- 失败条件使用相对参考根高度差 0.25 m 或根重力 z 差 0.8；MPJPE 为相对根部的身体位置误差（mm）。
- `sim.py` 固定摩擦 1.0，关闭自碰撞，不添加观测噪声或随机化。
- **开源限制：** 此仓库有回放/推理代码，但未提供训练入口，不能把它写成可复现作者训练的完整 pipeline。
- **执行限制：** 未实际运行 `cartwheel-play`、下载权重或执行真机测试；对源码的读取不等价于实验复现。

## 对 wiki 的映射

- [Cartwheel Control 主节点](../../wiki/entities/cartwheel-control.md)
- [官方模型卡归档](../sites/cartwheel-control-huggingface.md)
