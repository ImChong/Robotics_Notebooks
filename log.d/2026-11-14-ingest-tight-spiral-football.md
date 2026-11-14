# 2026-11-14 · ingest · 紧密螺旋橄榄球投掷

- **意图：** 收录爱荷华州立大学 Zaid Mahboob / Bowen Weng 关于 Unitree G1 精确旋转出球的 Humanoids 2026 论文，形成跨动态操作与接触控制的知识导航。
- **原文：** https://arxiv.org/html/2608.16642v2（已核对 §I–§V 方法、实验、局限）；作者项目页 https://www.zaidmahboob.com/projects/tight-spiral/
- **关键结论：** 两阶段抛投/随动；MPC 采样优化求解一次约 500 ms，真机使用离线 52 维查表；最高旋转效率 93.6%、最佳球头角 0.286 rad；不能误读为在线 250 Hz MPC 求解。
- **开放程度：** 论文和演示公开视频；未查到该论文的官方可运行控制器仓库及查表数据，故源码运行时序图明确不适用。
- **知识落点：** 建立唯一论文 wiki 实体、论文与项目页 sources 归档，加入 WBC 概念入链；包含原文数据/控制流程 Mermaid。
- **验证边界：** 按仓库来源与阅读骨架静态核对；通过 GitHub 连接器提交，当前容器无法访问 GitHub，未本地执行全量 make ci-preflight 或真机复现。
