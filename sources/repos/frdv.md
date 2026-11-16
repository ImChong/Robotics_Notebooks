# DARoSLab/FRDV：官方运动学重定向代码（2.0）

> 来源归档（repository / executable research code）

- **名称：** FRDV 2.0
- **仓库：** <https://github.com/DARoSLab/FRDV>
- **许可：** MIT（代码；数据、SMPL+H 与机器人模型各按其原始许可）
- **版本：** `pyproject.toml` = `2.0.0`；核对日期 2026-11-12
- **论文：** <https://arxiv.org/abs/2610.28470>
- **项目页：** <https://daroslab.github.io/frdv/>
- **核对范围：** README 全文、`pyproject.toml` 命令入口、`src/frdv/cli/retarget.py`、`src/frdv/cli/config.py`、`src/frdv/retarget.py`、`src/frdv/sqp.py`、`src/frdv/qp.py`、`src/frdv/parallel.py`、`src/frdv/_native.py`、`benchmarks/benchmark_nov_8_2026.md`；读源码未安装或执行。 
- **通读：** 已完成

## 一句话说明与开源边界

官方目前可以运行 **LAFAN1 / AMASS → Unitree G1 的约束运动学重定向、NPZ 保存及 Viser 回放**；**尚未包含论文中的 CI-iLQR 动力学验证管线**。这属于部分开源，不应把它标记为已完整开源论文端到端算法。

## 入口与关键实现

| 层级 | 文件 / 入口 | 已核对行为 |
|---|---|---|
| 配置与 CLI | `pyproject.toml`、`src/frdv/cli/config.py` | `retarget` 映射至 `frdv.cli.retarget:main`，Hydra 配置合成 |
| 数据加载 | `src/frdv/cli/retarget.py` `prepare`、`src/frdv/motion/` | 加载 BVH/AMASS，身体缩放及机器人关键点 targets |
| 运动学模型 | `src/frdv/retarget.py` | `IKProblem` 结合 tracking、regularization、smoothing 与关节/地面/可选自碰撞约束 |
| SQP / QP | `src/frdv/sqp.py`、`src/frdv/qp.py` | Gauss–Newton SQP；**2.0 QP 为 Goldfarb–Idnani 主动集**，有热启动 |
| 原生加速 | `src/frdv/_native.py`、`src/frdv/_native.c` | 编译并缓存 C 代码；整段求解在 ctypes 调用时释放 GIL |
| 调度 | `src/frdv/cli/retarget.py` | 多文件默认 `mode=files`，`ThreadPoolExecutor` 并行；缺原生路径时回退 Numba／逐帧方案 |
| 可选分块 | `src/frdv/parallel.py` | 显式 `mode=chunks`，边界 `handoff` 重算以避免线性混合造成不合理姿态 |
| 保存与回放 | `src/frdv/cli/retarget.py` `_save`；`play` CLI | `qpos` / `fps` / targets / metadata 保存到 NPZ，`frdv_config.yaml` 保存配置 |

## 官方最短运行路径

```bash
uv sync --locked --extra fast --extra cli --extra vis
uv run retarget robot=g1 dataset=lafan1 data=assets/mocap/sample/walk3_subject5.bvh end=300 out=outputs/quickstart
uv run play outputs/quickstart
```

G1 的 `qpos` 布局为根平移三维、根旋转四元数、29 个关节角，共 36 维。`uv run check` 可检查映射与跟踪误差。AMASS 的 SMPL+H 模型和完整动作数据均须另行从官方渠道获取。

## 性能测量口径（官方 2026-11-08）

在 Apple M1 8 核、8 GiB，LAFAN1 77 个动作（496,672 帧）条件下：

- FRDV 2.0，8 **file workers**：预热后**处理** 0.987 s、**处理+保存** 1.079 s；处理吞吐 503,288.6 FPS。
- FRDV 2.0，1 worker：预热后处理 4.725 s，吞吐 105,126.8 FPS。
- 8-worker 的首次完整流程 3.750 s；**包含必要编译及其他冷启动开销**（不含安装/下载）。
- `Warm FPS` 按 `496672 / Warm processing seconds` 计算；预热处理包含数据加载与预处理、重定向及可选清理，**不包含保存、共享初始化与编译**。
- 这些是**聚合离线吞吐**，不是机器人的在线控制频率，也不是论文动力学验证的运行速度。

基准原文件：<https://github.com/DARoSLab/FRDV/blob/main/benchmarks/benchmark_nov_8_2026.md>。

## 对 wiki 的映射

- [FRDV 主节点](../../wiki/entities/paper-fast-motion-retargeting-pipeline-humanoid.md)
- [论文归档](../papers/fast-motion-retargeting-pipeline-humanoid_arxiv_2610_28470.md)
- [项目主页归档](../sites/frdv-project.md)
