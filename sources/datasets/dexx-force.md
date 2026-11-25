# Dex-X Force｜带接触力标签的灵巧操作仿真轨迹

- **类型：** dataset
- **发布者：** Ruoqu Chen / Dex-X 团队
- **数据卡：** <https://huggingface.co/datasets/Ruoqu/DexX-Force>
- **固定版本：** `248e2192d8ac9f346d8397595f5c34e824da3e2a`（Hub API 记录最后更新于 2026-11-24 UTC）
- **对应论文：** [Dex-X，arXiv:2609.07747v3](https://arxiv.org/abs/2609.07747v3) · [论文归档](../papers/dex-x_arxiv_2609_07747.md)
- **下载和字段：** [LOADING](https://huggingface.co/datasets/Ruoqu/DexX-Force/blob/248e2192d8ac9f346d8397595f5c34e824da3e2a/LOADING.md) · [FORMAT](https://huggingface.co/datasets/Ruoqu/DexX-Force/blob/248e2192d8ac9f346d8397595f5c34e824da3e2a/docs/FORMAT.md)
- **入库日期：** 2026-11-25
- **核对范围：** 固定版本数据卡、加载说明、字段说明、来源与许可、目录结构、统计及发布校验报告；双来源 episode manifest、6 个样本的数组、官方 reader 直读 DexYCB tar 的路径与字段；参考论文 v3 §3–4 及触觉附录用于确认评测与模态边界。
- **通读：** 已完成

## 内容与使用边界

- 公开部分是 Hand–Arm Grasp & Lift-10K。Dexonomy 和经过 SPIDER 处理的 DexYCB 参考用于构建任务；保存的是 FR3 + 右 Sharpa 策略在仿真中的执行结果，而非源参考、真实传感器数据或论文所有任务。
- 双 manifest 字段实查：Dexonomy 为 10,028 条 / 2,954 个参考 / 767 个源物体，DexYCB 为 315 条 / 82 个参考 / 19 个归一化 YCB 身份。DexYCB `object_id` 在清单中按序列区分 82 个条目，不能当作 82 种物体。分源口径合计 786，不代表跨来源全局去重。
- 作者统计：13.14 小时、1,429,216 状态帧、1,418,873 动作帧；约 30 Hz 记录、120 Hz 物理仿真。作者收录口径是完整执行、最终误差低于 3 cm / 30°、抬升超过 3 cm 且未记录到抬升后掉落；不是成功率。
- 压缩 NPZ 的目标物体力是 `tactile_object_force_w`（5×3、牛顿、世界对齐轴）；全碰撞体力另有世界轴与掌心轴字段。用有效掩码读接触位置，四元数为 wxyz。预动作策略力缓存可能经过平滑和延迟。
- 路径：`Dataset.rows()` 读 manifest，`Dataset.rollout()` 按索引从 tar（或已落盘文件）取 NPZ，仅解码请求字段；`state[k]`、`action[k]`、`state[k+1]` 依次对齐。公开样例与一条分片内样本均实读，不等于重放、训练或全量质量验证。
- 无作者定义的训练 / 验证 / 测试拆分。`strict_grasp` 是发布索引的分组；按父参考及归一化物体身份做独立拆分。Dexonomy 对象几何须从上游另取。
- [许可与来源](https://huggingface.co/datasets/Ruoqu/DexX-Force/blob/248e2192d8ac9f346d8397595f5c34e824da3e2a/docs/SOURCES.md)：Hub 标记 `other` / 继承上游条款，Dexonomy 和 DexYCB 来源资料为 CC BY-NC 4.0，机器人、物体和源模型分别保留各自声明，勿套用代码仓的 MIT 到整个数据包。

## 对 wiki 的映射

- [Dex-X Force 数据节点](../../wiki/entities/dataset-dexx-force.md)
- [Dex-X 方法论文](../../wiki/entities/paper-dex-x.md)
