# SRL-MPC 官方项目页及补充评测

- **类型：** site
- **可访问链接：** <https://hku-sail.github.io/srl_mpc/>（2026-11-25 核对 HTTP 200）
- **论文所列旧链接：** <https://hanruihua.github.io/srl_mpc_project/>（本次核查 HTTP 404）
- **论文：** [HTML](https://arxiv.org/html/2608.21175v1) · [归档](../papers/srl_mpc_arxiv_2608_21175.md)
- **官方仓库：** [hku-sail/srl_mpc](https://github.com/hku-sail/srl_mpc) · [仓库归档](../repos/srl-mpc.md)

项目页给出三条独立训练策略的密集场景结果（每策略每设置 100 局）：20 台 91.0% ±2.6 pp，25 台 86.7% ±0.6 pp，误差为跨训练种子的样本标准差。站点另列五组不重新训练的 OOD 补充测试，宏平均成功率 77.5% ±2.5 pp；分项由[官方热图](https://hku-sail.github.io/srl_mpc/assets/results/ood-success-heatmap.svg)给出。真机为受控室内演示，不是数值仿真结果的真机复现实验。

项目站点按钮仍标注代码待发布。官方仓存在不代表算法开源；以[固定快照文件树](https://github.com/hku-sail/srl_mpc/tree/2d3d17bcc656a215f143b3839cb3544ab7231b26)核对公开内容。

- **对 wiki 的映射：** [SRL-MPC](../../wiki/entities/paper-srl-mpc.md)
