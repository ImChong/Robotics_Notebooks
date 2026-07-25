# mit-biomimetics/Cheetah-Software

> 来源归档

- **标题：** Cheetah-Software — MIT Biomimetic Robotics Lab 四足机器人与仿真控制栈
- **类型：** repo
- **作者 / 组织：** MIT Biomimetic Robotics Lab（mit-biomimetics）
- **链接：** https://github.com/mit-biomimetics/Cheetah-Software
- **许可：** MIT
- **星标（截至 2026-07-25）：** ~3262
- **真机运行文档：** https://github.com/mit-biomimetics/Cheetah-Software/blob/master/documentation/running_mini_cheetah.md
- **入库日期：** 2026-07-25
- **一句话说明：** Mini Cheetah / Cheetah 3 官方公开控制软件：公共动力学库、robot 程序、Qt 仿真、LCM 类型、user 控制器（含 MPC/WBC 等）。
- **开源状态：** **已开源**（软件栈；整机机械 CAD 不在本仓）
- **关联策展：** [mit_mini_cheetah_learning_stack_curator](../personal/mit_mini_cheetah_learning_stack_curator.md)
- **沉淀到 wiki：** [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)

---

## 仓库结构（README）

| 目录 | 作用 |
|------|------|
| `common/` | 动力学与工具公共库（含测试） |
| `robot/` | 真机程序 |
| `sim/` | Qt 仿真（唯一依赖 Qt 的程序） |
| `user/` | 用户控制器（如 `JPos_Controller`） |
| `lcm-types/` | LCM 消息类型 |
| `documentation/` | 含 `running_mini_cheetah.md` |
| `resources/` | 可视化用 CAD 等数据 |

## 构建要点

- 仿真 / 通用：`cmake .. && make`
- 面向 Mini Cheetah 交叉/部署：`cmake -DMINI_CHEETAH_BUILD=TRUE`
- 依赖：Qt 5.10、LCM、Eigen、mesa/freeglut、BLAS/LAPACK；可选 Ipopt

## 对 wiki 的映射

- [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)
- [srbd-convex-mpc-wbc](../../wiki/concepts/srbd-convex-mpc-wbc.md)
- [mpc-wbc-integration](../../wiki/concepts/mpc-wbc-integration.md)
