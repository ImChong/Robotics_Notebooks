## [2026-11-21] ingest | Zane Hub #398 机器人结构静刚度与末端变形预算

- 来源：sources/blogs/wechat_zanehub_robot_component_stiffness_deformation_2026-11-21.md；用户提供全文，微信原页未取得。
- 新增：wiki/concepts/robot-structural-static-stiffness-deformation-check.md，连接人形硬件设计纵深 Stage 2。
- 方法：以 TCP 工艺允许偏差为起点，分配连杆/关节/结合面静变形预算，经过梁手算、FEA、雅可比柔度映射、实测闭环。
- 勘误：按文章给定数值，I=1.12×10⁶ mm⁴、L=800 mm 挠度≈0.10884 mm、L=1200 mm≈0.36735 mm，均与原文不符；ISO 9283 静柔顺度与定位重复性分开对待。
- 证据边界：行业经验阈值、减速器参数、结合面 30% 与补偿 50% 等未逐条验证，不作为通用验收标准。
