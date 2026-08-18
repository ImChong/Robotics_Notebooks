# go2_flip_TO（yusongmin1/go2_flip_TO）

> 来源归档

- **标题：** go2_flip_TO
- **类型：** repo
- **来源：** 在 LAR SE3_TrajOpt 之上的社区扩展（维护者含 Evangelos Tsiatsianas、Konstantinos Chatzilygeroudis、yusongmin）
- **链接：** <https://github.com/yusongmin1/go2_flip_TO>
- **许可：** BSD-2-Clause
- **论文：** 同 [se3_trajopt](./se3_trajopt.md) / arXiv:2508.11520
- **项目页：** <https://lar.upatras.gr/projects/ibrics.html>
- **入库日期：** 2026-08-18
- **一句话说明：** SE(3) 切空间 TO 生成 Go2 前/后/侧空翻与行走轨迹，并写成 50 Hz AMP 风格 txt，供下游模仿 / 跟踪 RL。
- **沉淀到 wiki：** [`wiki/entities/paper-se3-tangent-to.md`](../../wiki/entities/paper-se3-tangent-to.md)

---

## 相对官方仓多出来的部分

| 项 | 说明 |
|----|------|
| 定位 | GitHub description「go2 后空翻脚本生成」；**未**标记为 `upatras-lar/se3_trajopt` 的 fork |
| PYTHONPATH | 必须 `$(pwd)/src/nltrajopt:$(pwd)/src`；只设 `src` 会 `ModuleNotFoundError: trajectory_optimization` |
| 线性求解器 | conda-forge cyipopt / IPOPT + **MUMPS**；文档明确不要 HSL `libhsl.so` |
| 敏捷入口 | `src/examples/agile_exps/quad_backflip.py`、`quad_frontflip.py`、`quad_sideflip.py`、`quad_sideflip_right.py`、`quad_jump_forward.py`、`quad_walk_forward.py` 等 |
| 导出 | `_export_go2_datasets.py` → `datasets/go2/mocap_motions_go2/<name>_50hz.txt`（JSON，每行 49 浮点） |
| 回放 | `datasets/viz_go2_amp_trajectory.py --amp ...`（MuJoCo）；求解过程可视化仍用 MeshCat `--vis` |
| 环境变量 | `GO2_NO_DATASET=1` 跳过写文件；`GO2_EXPORT_BASE_Z_OFFSET` 默认 0.022 m |

AMP 行布局：`root_pos(3) + root_rot xyzw(4) + dof_pos(12) + key_body_rel(12) + root_lin_vel(3) + root_ang_vel(3) + dof_vel(12)`；关节顺序 FL→RR 的 hip/thigh/calf。

## 最短复现（README）

```bash
conda create -n se3traj python=3.11 -y && conda activate se3traj
conda install -c conda-forge pinocchio cyipopt meshcat-python matplotlib numpy -y
export PYTHONPATH="$(pwd)/src/nltrajopt:$(pwd)/src"
python src/examples/agile_exps/quad_backflip.py --vis
# 回放导出：
export PYTHONPATH="$(pwd):$(pwd)/src/nltrajopt:$(pwd)/src"
python datasets/viz_go2_amp_trajectory.py --amp datasets/go2/mocap_motions_go2/quad_backflip_50hz.txt
```

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [se3_trajopt](./se3_trajopt.md) | 论文官方实现 |
| [paper-se3-tangent-to](../../wiki/entities/paper-se3-tangent-to.md) | 参数化对比与空翻结论 |
| [ahmp](./ahmp.md) | 同内核 + 接触发现 |
| [ibrics 项目页](../sites/ibrics-lar-upatras.md) | 演示与资助 |
