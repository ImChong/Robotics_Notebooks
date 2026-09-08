# TacPAC（LogosRoboticsGroup/TacPAC）

- **URL：** <https://github.com/LogosRoboticsGroup/TacPAC>
- **组织：** LogosRoboticsGroup（复旦大学 / SII Li Zhang 组）
- **许可证：** MIT
- **关联论文：** [tacpac_arxiv_2609_05266](../papers/tacpac_arxiv_2609_05266.md)
- **实体页：** [paper-tacpac](../../wiki/entities/paper-tacpac.md)

## 一句话说明

TacPAC 官方实现：两阶段 WAM + 触觉专家；`train_WanMoTJoint.sh` / `train_WanMoTJoint-TacExpert.sh` 训练，`deployment/model_server/server_infersystem.py` 提供 `predict_action` → `prefill_tactile_cache` → `correct_action` 闭环。Flexiv Rizon 4 真机部署；数据集与 checkpoint **待发布**。

## 交叉链接

- [TacPAC 论文实体](../../wiki/entities/paper-tacpac.md)
