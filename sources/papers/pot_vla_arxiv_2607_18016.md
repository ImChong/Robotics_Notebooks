# POT-VLA：Persistent 3D Object Tokens for Verifiable Loco-Manipulation（arXiv:2607.18016）

> 来源归档（ingest）

- **论文：** <https://arxiv.org/abs/2607.18016>（2026-07-20 提交）
- **作者：** Peng Ren, Haoyang Ge, Jiang Zhao, Cong Huang, Yukun Shi, Pei Chi, Kai Chen
- **代码/权重/数据：** 截至 2026-07-22，arXiv 未列官方项目页或公开资产
- **一句话说明：** 用 RGB-D 维护按角色索引的持久 3D 对象记录，让同一对象状态同时驱动 Unitree G1 全身动作与几何谓词验收。

## 核心摘录

- **问题：** 长时程人形移动操作中，遮挡、接触、移动和失败恢复会造成动作条件中的对象状态与验收状态分叉。
- **方法：** Persistent Object Tokenization（POT）把 RGB-D 观测转为持久、角色化的 3D 对象 token；POT-VLA 用同一记录生成动作并检查空间关系。
- **结果：** Unitree G1 八类实机任务中，匹配的 GR00T-N1.7 基线为 **39/80**，POT-VLA 为 **71/80**；Being-0 对齐任务为 **44/50**。

**对 wiki 的映射：** [`wiki/entities/paper-pot-vla.md`](../../wiki/entities/paper-pot-vla.md)、[VLA](../../wiki/methods/vla.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)。

