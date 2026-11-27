# AgentVLN：论文与补充材料核读

- **类型：** 会议论文
- **机构：** 南京航空航天大学、山东大学、浙江大学
- **预印本：** [arXiv:2603.17670v1](https://arxiv.org/abs/2603.17670v1)，2026-03-18；[PDF](https://arxiv.org/pdf/2603.17670v1)
- **正式版本：** [ECCV 2026](https://doi.org/10.1007/978-3-032-37167-6_7)，LNCS 17033，114–132 页，Springer 标注 2026-09-12 首次上线。
- **补充材料：** [作者公开的 13 页 PDF](https://github.com/Allenxinn/AgentVLN/blob/c1c63bd97d1c54c5694af3d61c63285001525439/assets/Supplementary_Materials.pdf)
- **项目页：** https://allenxinn.github.io/AgentVLN/
- **代码：** https://github.com/Allenxinn/AgentVLN
- **核对范围：** 通读公开预印本的相关工作、任务建模、全局和局部方法、训练、实验及消融，和官方补充材料的空间映射、训练数据、部署说明、定性轨迹及局限。会议引文与上线日期以出版社书目页为准。
- **通读：** 已完成

1. §3.2–3.4：由模型选择技能而非每步直接发速度；RGB-D 和位姿为 3D/2D 路点互转提供几何。全局导航以语义匹配确定路点，局部定位可请求额外距离线索。
2. §4 表 1–2：R2R-CE 未见场景 SR 67.2、SPL 64.7；RxR-CE SR 69.5、SPL 61.3、nDTW 74.6。数值取自仿真，基线观察模态并非都一致。
3. §4 表 3 与补充 §8：R2R 消融基线 SR 38.6，加入技能调度及空间映射后 59.7，加入细粒度纠偏后 65.6，最终 67.2。反光、突变光照等未被模拟充分覆盖，小模型在真机上仍会误触发技能。

- **对 wiki 的映射：** [AgentVLN](../../wiki/entities/paper-agentvln.md)
