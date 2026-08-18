# AHMP（Agile Humanoid Motion Planning）

> 来源归档

- **标题：** AHMP
- **类型：** repo
- **来源：** Inria HUCEBOT + 帕特雷大学 LAR
- **链接：** <https://github.com/hucebot/ahmp>
- **许可：** BSD-2-Clause
- **论文：** Humanoids 2025，DOI 10.1109/Humanoids65713.2025.11203211；HAL <https://hal.science/hal-05072261>
- **项目页：** <https://lar.upatras.gr/projects/ibrics.html>
- **内层 TO：** <https://github.com/upatras-lar/se3_trajopt>
- **入库日期：** 2026-08-18
- **一句话说明：** CEM-MD 接触发现 + SE(3) 切空间全身 TO 的开源实现；Docker 入口，Talos 扶手/烟囱实验脚本。
- **沉淀到 wiki：** [`wiki/entities/paper-ahmp.md`](../../wiki/entities/paper-ahmp.md)

---

## 仓库入口（README / `src/examples/cem_exps`）

| 组件 | 说明 |
|------|------|
| 安装 | `ci/` Docker：`docker build -t ahmp`；根目录 `./run_docker.sh` |
| 线性求解器 | 默认 IPOPT + HSL MA97（`ci/` 放 coinhsl）；可改其它线性求解器 |
| 并行评测 | `src/examples/cem_exps/trajopt_parallel.py --exp handrails\|chimney --robot talos` |
| 烟囱高度 | `--dz 1.0` 或 `3.0` |
| elite 消融 | `--abl 0.3\|0.5\|0.8` |
| 批量 | `run_exps.sh`（当前默认打开 chimney climb_low 10 次；扶手/高烟囱/消融多被注释） |
| 模块 | `src/cem/` 外层；`src/nltrajopt/` 内层 TO |

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-ahmp](../../wiki/entities/paper-ahmp.md) | 论文实体 |
| [se3_trajopt](./se3_trajopt.md) | README 指向的最新 TO 实现 |
| [go2_flip_to](./go2_flip_to.md) | 同 TO 内核的 Go2 空翻/AMP 导出扩展 |
| [ibrics 项目页](../sites/ibrics-lar-upatras.md) | 资助与演示入口 |
