# PyRoki（chungmin99/pyroki）

> 来源归档

- **标题：** PyRoki — Python Robot Kinematics Library
- **类型：** repo
- **组织：** UC Berkeley（Kim*, Yi* et al.）
- **主页：** <https://github.com/chungmin99/pyroki>
- **项目页：** <https://pyroki-toolkit.github.io/>
- **论文：** <https://arxiv.org/abs/2505.03728>
- **文档：** <https://chungmin99.github.io/pyroki/>
- **入库日期：** 2026-09-15
- **一句话说明：** JAX 模块化运动学优化库：可微 FK/碰撞、组合代价、jaxls 流形 LM；examples 覆盖 IK、轨迹优化、手/人形 retarget；被 ProtoMotions、KineBench 等引用为规划后端。
- **沉淀到 wiki：** [`wiki/entities/paper-notebook-pyroki.md`](../../wiki/entities/paper-notebook-pyroki.md)

---

## 开源状态

**已开源（截至 2026-09-15）**：`pip install -e .`（Python 3.10+）；`examples/` 含 14+ 脚本与 retarget 资产。

| 路径 | 角色 |
|------|------|
| `src/pyroki/` | 核心库（FK、碰撞、代价、求解接口） |
| `examples/01_basic_ik.py` … `14_singularity_aware_ik.py` | IK / 双臂 / 移动基座 / 碰撞 / 可操作度 / 在线规划 / TO |
| `examples/09_hand_retargeting.py` / `10_humanoid_retargeting.py` | 动捕→机器人重定向入门 |
| `examples/retarget_helpers/` | 重定向辅助脚本与数据 |
| `benchmark/` | 与 cuRobo 等对比实验脚本 |

---

## 推荐复现入口

```bash
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
python examples/01_basic_ik.py
python examples/10_humanoid_retargeting.py
```

依赖要点：**JAX**、[jaxls](https://github.com/brentyi/jaxls)、[jaxlie](https://github.com/brentyi/jaxlie)；URDF 机器人描述。

---

## 与仓库内实体的关系

- 论文实体：[paper-notebook-pyroki.md](../../wiki/entities/paper-notebook-pyroki.md)
- 对照：[curobo.md](../../wiki/entities/curobo.md)（GPU 运动生成；论文互有 benchmark）
- 下游：[protomotions.md](./protomotions.md)（v3 默认 retarget 后端）
- 评测引用：[kinebench.md](./kinebench.md)（规划 `pyroki` 模块）
