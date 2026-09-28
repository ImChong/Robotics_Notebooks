# lerobot-humanoid-identification

> 来源归档

- **标题：** lerobot-humanoid-identification（仓内目录 `identification_2`）
- **类型：** repo
- **链接：** https://github.com/Virgileboat/lerobot-humanoid-identification
- **许可证：** Apache-2.0
- **入库日期：** 2026-09-28
- **一句话说明：** LeRobot 双足 **关节级动力学参数辨识**：MJWarp 批量仿真回放 + **CMA-ES** 优化，缩小 sim–real 动力学差距；真机采集/延迟工具 intentionally out of scope。
- **代码：** https://github.com/Virgileboat/lerobot-humanoid-identification（**已开源**）
- **沉淀到 wiki：** [lerobot-humanoid](../../wiki/entities/lerobot-humanoid.md)

---

## 布局（README）

- `cmaes/` — 优化入口与目标函数
- `simulator/` — MJWarp 批量运行时/模型池
- `models/lerobot_humanoid/` — 常量与兼容导出
- `lerobot-humanoid-models/` — git submodule（MJCF）
- `results/baseline_controller_v1/...` — 参考辨识输出

## 快速开始

```bash
git submodule update --init --recursive
uv sync
```

示例数据集：`models/lerobot_humanoid/datasets/baseline_controller_v1`

## 与 runtime 分工

- 采集侧工具在 [lerobot-humanoid-runtime](lerobot_humanoid_runtime.md)（如 `tools/data_acquisition.py`）
- 本仓专注 **仿真参数辨识**

## 对 wiki 的映射

- [LeRobot Humanoid](../../wiki/entities/lerobot-humanoid.md)
