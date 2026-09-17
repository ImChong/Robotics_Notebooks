# RecMorph（arXiv:2609.18359）

> 来源归档（paper）

- **标题：** RecMorph: Topology-Guided Spatial Recurrence for Generalized Morphology Control
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.18359>
- **PDF：** <https://arxiv.org/pdf/2609.18359>
- **代码：** <https://github.com/quanruirao/RecMorph>
- **作者：** Quanrui Rao, Yong Liu, Xueming Xiao, Yingbo Luo, Kun Wu, Zhenyu Xu, Meibao Yao
- **入库日期：** 2026-09-17
- **一句话说明：** 运动学树 DFS 序 + 双向空间 RNN，在固定宽深下线性 token 复杂度实现跨形态 limb 通信与变换；UNIMAL 五任务均值最优，四足 Isaac Lab 共享策略 40 次 Go1/Go2 零 fall。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-09-17）：GitHub 含 UNIMAL + Isaac Lab 双栈训练/评测脚本与文档。

## 核心摘录

1. **问题：** 广义形态控制需跨 limb 信息变换、全身协调，且随 body size 保持效率；现有通信机制只部分满足。
2. **RecMorph：** 深度优先遍历将 kinematic tree 转为 **形态衍生序列**；共享双向 transition 沿序渐进变换 limb 信息再解码 action。
3. **稳定化：** residual preservation、RMS norm、input-dependent channel modulation。
4. **复杂度：** 固定模型宽深下 **线性 token 复杂度**。
5. **UNIMAL（5 任务）：** 评估的 generalized morphology controller 中 **mean final training performance 最强**；FT 上 **推理吞吐最高**；泛化至未见 variation 与 **30 limb** 机体。
6. **Isaac Lab 四平台：** Go1 / Go2 / ANYmal-B / ANYmal-C 共享策略；nominal + 高摩擦下 **macro-averaged performance 最佳**；nominal velocity RMSE 较 specialist MLP **↓43.5%**；**40** 次物理 Go1/Go2 trial **零 fall**。

**对 wiki 的映射**

- [paper-recmorph](../../wiki/entities/paper-recmorph.md)
