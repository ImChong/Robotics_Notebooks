# MuJoCo System Identification Toolbox（MuJoCo SysID）

> 来源归档（Google DeepMind 官方文档与代码；技术说明为归纳而非源码镜像）

- **标题：** System Identification Toolbox
- **类型：** repo（MuJoCo 主仓中的独立 Python 子模块）
- **维护方：** Google DeepMind（谷歌 DeepMind）
- **官方源码：** <https://github.com/google-deepmind/mujoco/tree/main/python/mujoco/sysid>
- **官方说明：** <https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/sysid/README.md>
- **官方 Python 文档：** <https://mujoco.readthedocs.io/en/stable/python.html#sysid>
- **官方 Notebook：** <https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/sysid/sysid.ipynb>
- **上游许可证：** Apache-2.0（源文件版权声明）
- **核对日期：** 2026-11-17；以检查当日 `main` 分支为准，API 可能演进
- **核对范围：** 完整阅读子目录 README；核对 `__init__.py` 导出 API；阅读 `_src/optimize.py`、`_src/residual.py`、`_src/trajectory.py` 的相关执行路径与 `_src/parameter.py`、`_src/model_modifier.py`、`_src/io.py`、`report/defaults.py` 的关键函数；检查 Notebook 的目录与安装入口。未执行 Notebook 全部单元。
- **通读：** 已完成
- **开源状态：** **已开源**（实现、Notebook、测试均在 MuJoCo 主仓可访问；未找到独立的 `mujoco-sysid` 官方子仓）
- **沉淀到 wiki：** [MuJoCo SysID](../../wiki/entities/mujoco-sysid.md)

## 官方设计与问题定义

给定 `MjSpec` 模型、真机控制序列与测量信号，联合估计让仿真输出靠近真机的模型/测量参数。默认构造成**带 box bounds 的非线性最小二乘**；MuJoCo 优化后端使用有限差分 Jacobian / Gauss–Newton 思路，而非对 MuJoCo 物理引擎自动微分。对不同参数扰动的 rollout 可打包，通过 `mujoco.rollout` 在多线程执行。

### 核心数据与可替换接口

| 组件 | 在仓库中的职责 |
|---|---|
| `Parameter` / `ParameterDict` | 定义初值、上下界、frozen 状态与修改 `MjSpec` 的回调 |
| `TimeSeries` | 控制、传感器、状态的时间轴与按名字对齐 |
| `ModelSequences` | 一份 `MjSpec` 和一个或多个实测轨迹；多个 group 可联合拟合 |
| `build_residual_fn` | 绑定数据与配置，返回 `(x, params) -> residuals` |
| `optimize` | 默认 `optimizer="mujoco"`，另有 `scipy` / `scipy_parallel_fd` |
| `save_results` / `default_report` | 保存参数、优化结果、置信区间、模型及可视化报告 |
| `build_model` / `custom_rollout` / `modify_residual` | 允许替换模型构建、仿真路径或残差计算 |

默认残差路径：`residual()` → `model_residual()` → 由 `Parameter` modifier 构建模型 → `sysid_rollout()` → `mujoco.rollout.rollout()` → 预测/实测时间对齐、`weighted_diff`、`normalize_residual`。

## 可辨识参数与工程注意

- 官方示例覆盖接触滑动摩擦、关节阻尼、连杆质量与质心、完整惯量，以及位置伺服执行器 P/D 增益。
- 完整惯量支持基于 **pseudo-inertia Cholesky** 的 10 维物理一致性参数化（官方 README 引 Rucker & Wensing, 2022）；传感器偏置、增益、时延也可进入后处理残差。
- `check_conditioning=True` 可以做初始 Jacobian 条件数诊断，但要额外计算一次有限差分 Jacobian。
- 默认 `n_threads` 来自 CPU 数量；每个差分扰动需要重新仿真，参数多/轨迹长时计算成本上升。
- `TimeSeries` 的传感器名字、顺序、时间戳和 `mjSTATE_FULLPHYSICS` 初态必须准确匹配模型。没有充分激励的数据会导致参数不可辨识，低训练误差不能直接证明真实闭环性能。
- `default_report` 允许生成 HTML 诊断、拟合曲线、参数表与置信区间；默认视频生成还要求渲染环境与 `save_path`。

## 复现入口（官方）

```bash
pip install 'mujoco[sysid]'
```

- [官方 README 端到端示例](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/sysid/README.md#example)：以 `MjSpec.from_file`、`ParameterDict`、`TimeSeries`、`ModelSequences` 组织一条轨迹，再调用 `build_residual_fn`、`optimize`、`save_results`。
- [Notebook 在线 Colab](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mujoco/sysid/sysid.ipynb)：目录列有弹簧阻尼质量辨识、5-DoF 机械臂 armature 辨识、可辨识性诊断；安装 cell 使用带 `--pre` 的官方试用安装命令。
- [优化源码](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/sysid/_src/optimize.py) · [残差源码](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/sysid/_src/residual.py) · [rollout 源码](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/sysid/_src/trajectory.py) · [报告源码](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/sysid/report/defaults.py)

## 对 wiki 的映射

- [MuJoCo SysID（工具实体）](../../wiki/entities/mujoco-sysid.md)
- [System Identification（方法概念）](../../wiki/concepts/system-identification.md)
- [MuJoCo（物理引擎）](../../wiki/entities/mujoco.md)
- [Sim2Real 纵深路线 Stage 1](../../roadmap/depth-sim2real.md)
