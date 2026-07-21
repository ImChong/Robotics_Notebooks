# shared-controllers（gbionics / ami-iit）

> 来源归档

- **标题：** Shared Controllers — whole-body control library for humanoid robots
- **类型：** repo（人形全身控制 / 共享自主框架）
- **机构：** GenerativeBionics（`gbionics`）；源自 IIT AMI 工作线
- **链接（当前）：** https://github.com/gbionics/shared-controllers
- **论文中的 URL：** https://github.com/ami-iit/shared-controllers（截至 2026-07-21 **重定向/迁移** 至上列 `gbionics` 仓）
- **论文：** https://doi.org/10.1038/s42256-026-01272-2
- **Stars：** ~4（2026-07）
- **入库日期：** 2026-07-21
- **许可证：** 公开仓页面未稳定展示 SPDX（以仓内 LICENSE 为准；入库时 API 返回 `null`）
- **代码 / 开源状态：** **已开源** — `lib/wholebodycontrollib` + `scripts/` 真机/仿真应用；依赖 YARP、iDynTree、可选 Gazebo/YARP 插件与 `bipedal-locomotion-framework`
- **一句话说明：** ergoCub「物理智能」实例与协作/行走实验脚本所依赖的 Python WBC 栈；复现需 iCub/YARP 生态。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-ergocub-shared-embodied-intelligence.md`](../../wiki/entities/paper-ergocub-shared-embodied-intelligence.md)
- **交叉归档：** [paper-sartore-2025-ergocub-nmi.md](./paper-sartore-2025-ergocub-nmi.md)、[ami-iit-adam.md](./ami-iit-adam.md)、[ergocub-eu.md](../sites/ergocub-eu.md)

---

## 仓内结构（2026-07 快照）

| 路径 | 作用 |
|------|------|
| `lib/` | `wholebodycontrollib` Python 库 |
| `scripts/` | 真机或仿真上跑 WBC 的应用入口 |

## 安装入口（README）

```bash
conda create -n sc_env python=3.10
conda activate sc_env
conda install -c conda-forge pip matplotlib qpsolvers resolve-robotics-uri-py yarp idyntree
# 可选仿真/真机：gazebo-yarp-plugins icub-models bipedal-locomotion-framework
git clone https://github.com/ami-iit/shared-controllers.git  # 或 gbionics/shared-controllers
cd shared-controllers && pip install .
```

## 对 wiki 的映射

- [Whole-Body Control](../../wiki/concepts/whole-body-control.md) — 分层轨迹生成 / 调整 / QP 控制的工程实例
- 实体页「源码运行时序图」物理智能支路
