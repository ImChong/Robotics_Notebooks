# 2026-11-13 ingest：Flash ALOHA（高速双臂动觉示教）

- **Op：** ingest
- **意图：** 为机器人学习知识库归档 Stanford Flash ALOHA 的原始论文、项目页与源码，并写入独立、可导航的论文详情节点。
- **查重：** 已用 Flash ALOHA、flash-aloha 与 ALOHA Lightning 查询当前仓库内容和 PR；未发现同论文专属详情页，新建唯一节点 `wiki/entities/paper-flash-aloha.md`。
- **原文核验：** 官方论文 PDF 12 页（方法、Table I–III、附录）与官方站点，来源完整记录于 `sources/papers/flash_aloha_icra2026.md`。
- **开源结论：** 官方 GitHub 已开放 MIT 许可的采集、训练及真机推理框架；未确认原论文完整五任务示范与检查点公开下载；新仓库 Flow/RTC 能力不可反推为论文主结果。
- **关键技术：** 可反驱手持示教、编码器采集轨迹、基于运动学的操作员手部图像遮罩、ACT 动作块重叠 Sigmoid 平滑。节点内含原文流程 Mermaid 和结合真实源文件入口的运行时序 Mermaid。
- **评测读法：** 各 50 条示范、五项成功率 80%–100%；电池插入 5.70 s vs 主从 17.87 s；抛接球 1.85 s。把速度、数据采集与任务成功率的因果边界分开。
- **关联入口：** `wiki/tasks/bimanual-manipulation.md`，避免孤立页。
- **源码边界：** 已追踪 `scripts/run.py` → 采集/训练/推理入口及 `blend_actions`；论文 FK+相机标定手部遮罩的完整函数仍未定位，不宣称源码完全复核。
- **校验：** 当前 GitHub 连接仅允许读写线上仓库，无法在此运行 `make ci-preflight` 完整本地检查；最终以 GitHub Actions 的 Wiki Lint / Mermaid 检查为准。
