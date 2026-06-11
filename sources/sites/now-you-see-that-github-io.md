# Now You See That 项目页（hellod035.github.io/Now_You_See_That）

> 来源归档（ingest 附属）

- **标题：** Now You See That: Learning End-to-End Humanoid Locomotion from Raw Pixels — Project Page
- **类型：** site / demo videos
- **链接：** <https://hellod035.github.io/Now_You_See_That/>
- **关联论文：** [now_you_see_that_arxiv_2602_06382.md](../papers/now_you_see_that_arxiv_2602_06382.md)（arXiv:2602.06382）
- **入库日期：** 2026-06-11
- **一句话说明：** RSS 2026 接收论文配套站：两阶段方法示意、8 步深度增广可视化、多样地形增广样例与 **Wild Parkour / 上下楼梯 / 垫脚石 / 室内跑酷 / 平衡恢复** 等实机视频。

## 页面结构（2026-06-11 抓取）

1. **Method Overview** — Stage 1 特权 height scan + 多 critic/discriminator RL；Stage 2 vision-aware 蒸馏到增广深度。
2. **Augmentation Pipeline** — 左右深度 → 立体融合 → 卷积 → 高斯 → Perlin → 尺度 → 像素失效 → 裁剪裁剪，逐步可视化。
3. **More Augmentation Results** — 多地形 triplet（左/右/增广输出）；深度归一化 [0, 2] m 色图。
4. **Real World Results** — 六类实机场景视频：
   - Wild Parkour
   - Stair Up / Stair Down
   - Step Stone
   - Indoor Parkour
   - Balance Recovery
5. **Abstract** — 与 arXiv 摘要一致。
6. **机构标签** — HIT + HONOR Robotics Team；RSS 2026 Accepted。

## 对 wiki 的映射

- 实体页演示索引：[paper-now-you-see-that-humanoid-vision-locomotion.md](../../wiki/entities/paper-now-you-see-that-humanoid-vision-locomotion.md)「推荐继续阅读」
