# OSQP

> 来源归档（Humanoid Motion Intelligence 开源项目主表）

- **标题：** OSQP
- **类型：** repo
- **技术路线分组：** 工程与实机部署
- **链接：** https://github.com/osqp/osqp
- **入库日期：** 2026-07-30（首次）；2026-11-07（README 与官方文档核对）
- **核对范围：** 读官方 README（问题形式 $\min \tfrac12 x^TPx+q^Tx$ s.t. $l\le Ax\le u$）、`docs/solver/index.rst`（polishing 段）、`docs/interfaces/solver_settings.rst`（`warm_starting` 默认 True、`eps_abs` 等设置）、`docs/interfaces/python.rst` 的 setup / update 说明；算法细节以配套论文通读归档为准（见 `sources/papers/osqp_admm_qp_arxiv_1711_08013.md`）。未读 C 源码，未运行基准。
- **通读：** 已完成
- **一句话说明：** 用算子分裂求解稀疏凸二次规划，并支持热启动、代码生成与多语言绑定；在人形控制中常承载接触力分配、约束逆动力学和MPC子问题，稳定的状态复用有利于实时循环。
- **开源状态（据主表）：** 已开源（以官方仓库 README 为准）
- **策展入口：** [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)
- **沉淀到 wiki：** 是 → [`wiki/entities/osqp.md`](../../wiki/entities/osqp.md)

## 为什么值得保留

主表将该项列为人形运动智能六条路线之一的工程/研究入口；本库为其建立独立详情节点，便于选型与交叉引用，而不镜像主表全文。

## 对 wiki 的映射

- [OSQP](../../wiki/entities/osqp.md)
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)

## 文档核对要点（2026-11-07）

- Python 接口：`m.setup(P=, q=, A=, l=, u=, **settings)` → `m.solve()`；参数化更新 `m.update(q=, l=, u=)`（仅向量，可复用分解）、`m.update(Px=, Ax=, ...)`（矩阵数值）；可手动 `m.warm_start(x=, y=)`。
- 默认设置（`solver_settings.rst`）：`eps_abs = 1e-3`、`max_iter = 4000`、`warm_starting = True`；`polish` 需显式开启，成功率随容差变小而升高，需额外一次分解。
