# ShowNow：用细粒度推测式流生成提升世界模型交互响应｜Fine-Grained Speculative Streaming for Responsive World Models

- **类型：** paper（arXiv 预印本）
- **作者：** Yiren Song、Guian Fang、Mike Zheng Shou
- **机构：** 新加坡国立大学 Show Lab（Show Lab, National University of Singapore）
- **版本：** arXiv:2611.34693v1，2026-11-25（UTC）
- **论文：** https://arxiv.org/abs/2611.34693
- **原文全文：** [HTML](https://arxiv.org/html/2611.34693v1) · [PDF](https://arxiv.org/pdf/2611.34693v1)
- **代码：** https://github.com/showlab/ShowNow
- **核对范围：** v1 正文及附录，重点对照 §3 树掩码与共享推理、§4 表 1–3、图 3、附录 A 计时、附录 G 画质与附录 H–I 范围；代码与资产单独归档。
- **通读：** 已完成
- **wiki：** [ShowNow 唯一详情](../../wiki/entities/paper-shownow.md)

## 核对锚点

- §3.2–3.4：匹配分支共享初始引擎状态和噪声，但每个未来的动作和相机条件各自独立；九个历史与十二个未来 latent、八分支的联合序列布局从 168 缩到 105。此计数不是全模型加速倍数。→ [核心原理](../../wiki/entities/paper-shownow.md#核心原理)
- §3.5、附录 C：复用未加位置变换的历史 K/V，窗口滑动时重加位置变换；按归一化因子合并公共历史和分支局部注意力；只有选中的干净 latent 进入持续流。→ [运行时序](../../wiki/entities/paper-shownow.md#源码运行时序图)
- §4、表 1、表 3：620 → 208 ms 是不同原生块、提前阈值、注意力版本和调度的完整部署对照；同学生的 16 fps 对照主要改善重启空帧。→ [评测边界](../../wiki/entities/paper-shownow.md#实验与评测)
- 附录 A、G、H、I：计划展示与可辨识动作效果分开计时；有长期画面漂移与中间命令被覆盖的风险；33B 的解码耗时为混合计时模型。→ [局限](../../wiki/entities/paper-shownow.md#局限与风险)
