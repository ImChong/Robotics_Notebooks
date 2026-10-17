# R²-WAM 项目页（kaipengfang.github.io/r2-wam）

> 来源归档

- **标题：** R²-WAM: Data-Free Continual Learning for World Action Models
- **类型：** site（项目页）
- **URL：** <https://kaipengfang.github.io/r2-wam/>
- **论文：** <https://arxiv.org/abs/2610.18038> — 归档见 [`sources/papers/r2_wam_arxiv_2610_18038.md`](../papers/r2_wam_arxiv_2610_18038.md)
- **代码：** 页面标注 "Code (coming soon)"，无 GitHub / Hugging Face 链接（2026-10-17 核查）
- **入库日期：** 2026-10-17
- **一句话说明：** 展示整流生成回放的方法、LIBERO 与真机结果、消融与回放预算。

## 页面要点（2026-10-17 核查）

| 区块 | 内容 |
|------|------|
| 作者与单位 | 同 arXiv；UESTC、Shanghai AI Lab、复旦、上海交大、同济、上海创智学院 |
| 方法 | 重生成旧任务 rollout，VLM 截断 + 光流一致性过滤后与新任务示范一起回放 |
| LIBERO | R²-WAM 92.7%；经验回放 94.6%；原始生成回放 70.4%；顺序微调 12.5%；NBT 2.33 对 80.19 |
| 真机 | R²-WAM 89.5%，ER 91.8%，顺序微调 28.5% |
| 消融 | 完整 94.0%；无过滤 80.8%；无截断 70.8% |
| 回放预算 | 每个旧任务 20 条整流 rollout |
