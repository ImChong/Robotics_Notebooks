# 2026-11-13 ingest：Mundo 相机 Plücker 射线与可编程仿真技术精读

- **意图：** 对已有 Mundo 节点增量深入整理，避免重复实体；厘清 Scene Program、MuJoCo、Proxy Video、Plücker Camera Encoder 与视频扩散 Transformer 的职责分界。
- **原始资料：** 2026-10-23 作者官方技术博客全文、项目主页全部技术板块、商汤 2026-11-04 官方新闻、Luo-Yihang/Mundo 仓库 README；数学定义参考 ICLR 2024 Cameras as Rays。
- **核心修订：** 增加 Plücker 射线定义与相机坐标约定、注入位置、无位姿参考图的 learned embedding；改绘项目 Mermaid 方法图，补输入/仿真/训练/性能边界。
- **开放结论：** 2026-11-13 官方仓库只有 README 与 assets，README 将代码/权重列为 11 月底待发布；不绘制虚构源码运行时序图。
- **质量：** 为原文归档补显式通读记录，wiki 增加 original 和统一阅读骨架；仓库完整预检需以 GitHub Actions 为准。
