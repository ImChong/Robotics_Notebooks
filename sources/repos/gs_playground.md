# gs_playground

> 来源归档

- **标题：** GS-Playground
- **类型：** repo
- **来源：** discoverse-dev（GitHub 组织）
- **链接：** https://github.com/discoverse-dev/gs_playground
- **入库日期：** 2026-04-29
- **一句话说明：** 将并行物理仿真与批量 3D Gaussian Splatting (3DGS) 渲染耦合，实现最高 10^4 FPS 光真实感视觉观测，用于视觉强化学习，已被 RSS 2026 收录。
- **沉淀到 wiki：** 是 → [`wiki/entities/gs-playground.md`](../../wiki/entities/gs-playground.md)

---

## 核心定位

GS-Playground 解决了视觉 RL 训练的两个核心矛盾：
1. **吞吐量 vs 真实感**：传统渲染（光栅化/光追）速度慢；GS-Playground 用批量 3DGS 渲染实现两者兼顾
2. **Sim2Real 外观迁移**：3DGS 场景由真实环境重建，天然缩小 visual sim-to-real gap

会议：**RSS 2026**（Robotics: Science and Systems）

---

## 技术架构

### 1. 渲染引擎：批量 3DGS

| 指标 | 数值 |
|------|------|
| 渲染分辨率 | 640×480 |
| 吞吐量 | 最高 **10,000 FPS** |
| 输出 | RGB + 深度（Depth）观测 |
| 技术 | Batch 3D Gaussian Splatting |

**Rigid-Link Gaussian Kinematics**：将 3DGS 高斯团簇绑定到物理刚体上，保证物体运动时渲染与物理同步。

### 2. 物理引擎

- **速度冲量求解器（Velocity-Impulse Solver）**：适合接触丰富（contact-rich）任务的稳定模拟
- 支持四足、人形、机械臂等多种机体

### 3. Real2Sim 工作流

从真实场景采集 → 3DGS 重建 → 直接导入仿真，实现外观级别的 Sim2Real 对齐。

---

## 仓库结构（早期预览版）

```
gs_playground/
├── benchmark/           # 批量渲染 Benchmark notebook
├── demo/
│   ├── live_demo/       # Franka / Robotiq 重放 demo
│   └── navigation/      # Go1 / Go2 / G1 locomotion demo
├── media/               # 资产与文档图片
└── pyproject.toml       # 依赖（uv sync 安装）
```

当前为早期预览（19 commits）；训练 pipeline、benchmark 套件、Real2Sim 工具将在后续版本发布。

---

## 与本项目其他资料的关系

| 资料 | 关系 |
|------|------|
| [isaac_gym_isaac_lab.md](isaac_gym_isaac_lab.md) | 同为并行仿真框架，IsaacLab 用光栅化渲染，GS-Playground 用 3DGS |
| [mujoco.md](mujoco.md) | 物理内核参考，GS-Playground 使用自研速度冲量求解器 |
| [sim2real.md](../../sources/papers/sim2real.md) | Real2Sim 工作流直接服务 sim2real 外观对齐 |
| [genesis-sim.md](../../wiki/entities/genesis-sim.md) | Genesis 也强调高吞吐仿真，但侧重多物理场，GS-Playground 侧重光真实感视觉 |
