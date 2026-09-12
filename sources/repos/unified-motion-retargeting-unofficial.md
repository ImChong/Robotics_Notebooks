# Unified-Motion-Retargeting（社区非官方复现）

> 来源归档

- **标题：** Unified-Motion-Retargeting（UMR 非官方复现）
- **类型：** repo（community / unofficial）
- **链接：** <https://github.com/longchengzhuo/Unified-Motion-Retargeting>
- **论文：** [arXiv:2609.02134](https://arxiv.org/abs/2609.02134)（与官方论文相同；**非作者维护**）
- **许可：** MIT
- **入库日期：** 2026-09-12
- **一句话说明：** 独立复现 UMR 两阶段管线：MuJoCo 3.9 + [mink](https://github.com/kevinzakka/mink) + Clarabel；默认 Unitree G1（29 DoF）+ BVH 样例；**不等同于官方 hanyang9/UMR**。
- **沉淀到 wiki：** 交叉引用 → [`wiki/entities/paper-umr-unified-motion-retargeting.md`](../../wiki/entities/paper-umr-unified-motion-retargeting.md)「工程实践 / 局限」

## 摘录要点

- README 明确标注 **Unofficial**，与作者无隶属关系。
- 栈：`environment.yml`（Python 3.10）、`scripts/retarget.py`（三阶段 + live player）、`scripts/report.py`（动力学检查 + 指标）。
- 模块：`umr/correspondence/`（Stage I）、`umr/tasks/` + `umr/limits/`（mink Task/Limit）、`umr/retarget/`（逐帧 + pkl 导出）。
- 默认机器人：`assets/robots/g1_description/`；样例 `data/walk_slow.bvh`。
- 输出：`output/<human>_to_<robot>/`（motion、video、report）。

## 与官方仓差异（选型读法）

| 维度 | 官方 [hanyang9/UMR](umr.md) | 本仓 |
|------|------------------------------|------|
| 维护 | 作者团队 | 社区个人 |
| 源覆盖 | SMPL-X、SOMA、GRAIL、OmniContact、OMOMO、MimicKit、AdaPT、NR 等 | 当前以 BVH + G1 为主 |
| Studio / 批处理 | 有 Studio + batch DP warm start | 无 Studio；单仓 retarget/report |
| 复现背书 | 论文实验与真机管线 | 学习/对照用；指标不与论文表直接等价 |

## 对 wiki 的映射

- 主实体：[paper-umr-unified-motion-retargeting.md](../../wiki/entities/paper-umr-unified-motion-retargeting.md)
- 官方代码：[umr.md](umr.md)
