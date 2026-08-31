# hanbyelcho.info/safeflow（项目页）

> 来源归档（ingest）

- **标题：** SafeFlow: Real-Time Text-Driven Humanoid Whole-Body Control via Physics-Guided Rectified Flow and Selective Safety Gating
- **类型：** site / project-page
- **官方入口：** <https://hanbyelcho.info/safeflow/>
- **入库日期：** 2026-08-31
- **一句话说明：** 三星 Future Robot AI Group 配套站点：物理引导整流流 + 三阶段安全门方法图、相对 TextOp 的量化对比、Unitree G1 真机长时域演示；截至 2026-08-31 未列官方代码仓库。

## 页面公开信息（检索自 2026-08-31）

| 资源 | URL / 状态 |
|------|------------|
| 项目首页 | <https://hanbyelcho.info/safeflow/> |
| arXiv | <https://arxiv.org/abs/2603.23983> |
| arXiv HTML | <https://arxiv.org/html/2603.23983> |
| **代码** | **未开源** — 页面无 GitHub / Hugging Face / 权重下载链接 |

## 与论文一致的公开主张（便于 wiki 溯源）

1. **两层架构：** 高层 physics-guided rectified flow（VAE 潜空间 + Reflow，NFE=1）生成参考轨迹；低层 RL 运动跟踪控制器执行；部署时三阶段安全门选择性放行。
2. **三阶段安全门：** (1) 文本嵌入 Mahalanobis 语义 OOD；(2) 方向敏感性差异 \(\mathcal{R}\) 检测生成不稳定并触发站立 fallback；(3) 关节/速度硬约束筛查。
3. **相对 TextOp：** 关节越界率 43.14%→3.08%；系统成功率 80.6%→98.5%；完整管线约 67.7 Hz（生成器 92.6 Hz）。
4. **真机：** Unitree G1 连续多行为流式文本控制；OOD 高风险指令（如 double backflip）被拦截。

## 对 wiki 的映射

- [`wiki/entities/paper-loco-manip-161-104-safeflow.md`](../../wiki/entities/paper-loco-manip-161-104-safeflow.md)
- [`sources/papers/safeflow_arxiv_2603_23983.md`](../papers/safeflow_arxiv_2603_23983.md)
