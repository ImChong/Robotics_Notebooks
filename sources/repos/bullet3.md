# Bullet3 Physics SDK

> 来源归档

- **标题：** Bullet Physics SDK (bullet3)
- **类型：** repo
- **来源：** Erwin Coumans 等（Bullet Physics 社区）
- **链接：** https://github.com/bulletphysics/bullet3
- **Stars：** ~14.5k（2026-06）
- **入库日期：** 2026-06-17
- **一句话说明：** 官方 **C++ 实时碰撞检测与多物理仿真 SDK**：刚体、软体/FEM、VR 与游戏向示例；**PyBullet Python 绑定与 `pybullet_envs` 均在本仓 `examples/pybullet`**，机器人 RL 领域通常经 PyBullet 而非直接链接 C++ API。
- **沉淀到 wiki：** [PyBullet](../../wiki/entities/pybullet.md)

---

## 核心定位

- **实时物理**：碰撞检测 + 刚体动力学；可选 **OpenCL GPGPU** 将碰撞与动力学整段跑在 GPU（需高端独显，实验性质）。
- **多场景**：游戏/VFX、VR sandbox（HTC Vive / Oculus）、机器人学与机器学习。
- **许可**：zlib（宽松开源）。
- **构建**：CMake / premake；Windows / Linux / macOS / iOS / Android；`vcpkg install bullet3` 亦可。

## 与 PyBullet 的关系

| 层级 | 仓库 / 入口 | 典型用户 |
|------|-------------|----------|
| C++ 核心 | `bulletphysics/bullet3` | 游戏引擎集成、ExampleBrowser、VR 沙盒 |
| Python 绑定 | 同仓 `examples/pybullet`；`pip install pybullet` | 机器人 RL、课程、快速 URDF 闭环 |
| 官方站点 | [pybullet.org](https://pybullet.org/wordpress/) | Quickstart、论坛、研究案例索引 |

README **明确推荐**机器人 / RL / VR 场景优先使用 **PyBullet** 而非裸 C++ API；安装示例：

```bash
pip3 install pybullet --upgrade --user
python3 -m pybullet_envs.examples.enjoy_TF_AntBulletEnv_v0_2017may
python3 -m pybullet_envs.examples.enjoy_TF_HumanoidFlagrunHarderBulletEnv_v1_2017jul
python3 -m pybullet_envs.deep_mimic.testrl --arg_file run_humanoid3d_backflip_args.txt
```

## 对 wiki 的映射

- [PyBullet（实体页）](../../wiki/entities/pybullet.md) — Python 封装层、URDF 闭环与 RL 生态
- [motion_imitation（四足模仿）](../../wiki/entities/motion-imitation-quadruped.md) — Google Research `motion_imitation` 基于 PyBullet
- [gym-pybullet-drones](../../wiki/entities/gym-pybullet-drones.md) — 四旋翼 Gymnasium 环境
- [contact-complementarity](../../wiki/formalizations/contact-complementarity.md) — Bullet 类引擎常用 hard contact / LCP 求解

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [pybullet-org.md](../sites/pybullet-org.md) | 官方 WordPress 站：研究案例、Colab、TDS 等 |
| [gym_pybullet_drones.md](gym_pybullet_drones.md) | 下游 RL 环境，依赖 PyBullet wheel |
| [motion_imitation_peng.md](motion_imitation_peng.md) | 四足模仿动物官方实现 |
