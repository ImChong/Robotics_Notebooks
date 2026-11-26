# RetinaGAN: An Object-aware Approach to Sim-to-Real Transfer · 方法页一手资料复核

- **原文：** <https://arxiv.org/html/2011.03148v2>
- **通读：** 已完成
- **核对范围：** 正文 §I–VI、Fig. 1 和附录中检测器训练、适配器训练、任务与评测方法；参考文献仅浏览。
- **入库日期：** 2026-11-26

以 CycleGAN 的非配对域转换为基础，用检测器的框与分类输出约束翻译后的物体一致性，采用焦点一致性损失。先适配模拟图，再训练行为策略。对抓取、推物和开门报告真机测试；“无新增真机数据”指相近视觉域的推物复用，不代表 RetinaGAN 从未使用真机图。

- [关联方法页](../../wiki/methods/domain-adaptation-for-sim2real.md)
