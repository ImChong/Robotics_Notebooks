# apollo-py（Apollo-Lab-Yale）

> 来源归档

- **标题：** apollo-toolbox-py / apollo-py
- **类型：** repo（Python 包；URDD 消费 + Blender 可视化）
- **代码：** <https://github.com/Apollo-Lab-Yale/apollo-py>
- **PyPI：** <https://pypi.org/project/apollo-toolbox-py/>（0.0.13，MIT）
- **入库日期：** 2026-05-17
- **最后更新：** 2026-08-29
- **一句话说明：** 根 **README** 仍几乎只有标题；`apollo_toolbox_py/` 同时承载 **URDD Python 消费**（链 / FK / 模块）与 **Blender 出图**（`apollo_py_blender.ChainBlender`）。APOLLO Blender 论文示例里的 `blender_robot_toolbox_py` 包名 **不在 PyPI**。
- **沉淀到 wiki：** [URDD 论文实体](../../wiki/entities/paper-urdd-universal-robot-description-directory.md)、[APOLLO Blender 论文实体](../../wiki/entities/paper-apollo-blender.md)

---

## 开源与可运行入口（2026-08-29 复核）

- **许可：** MIT（`LICENSE`）。
- **Blender 模块：** `apollo_toolbox_py/apollo_py_blender/`
  - `robotics/chain_blender.py` — `ChainBlender.spawn` / `set_state` / `keyframe_state` / `keyframe_discrete_trajectory`，以及 plain / 凸包 / 凸分解的可见性与配色
  - `viewport_visuals/lines.py`、`cubes.py` — 论文 `BlenderLineSet` / `BlenderCubeSet`
  - `scripts/test.py` — UR5 + `keyframe_discrete_trajectory` 最短脚本（`new_from_default_apollo_robots_dir()`）
- **可选依赖：** `pyproject.toml` extra `bpy` 钉 `bpy`/`easybpy`，且 **Python 3.11**；应装进 Blender 捆绑解释器。
- **资源目录：** 脚本读本机 URDD 根（默认 apollo robots dir），不是仓内自带网格。

---

## 对 wiki 的映射

- [`wiki/entities/paper-urdd-universal-robot-description-directory.md`](../../wiki/entities/paper-urdd-universal-robot-description-directory.md)
- [`wiki/entities/paper-apollo-blender.md`](../../wiki/entities/paper-apollo-blender.md)
- [`sources/papers/urdd_beyond_urdf_arxiv_2512_23135.md`](../papers/urdd_beyond_urdf_arxiv_2512_23135.md)
- [`sources/papers/apollo_blender_arxiv_2512_23103.md`](../papers/apollo_blender_arxiv_2512_23103.md)
