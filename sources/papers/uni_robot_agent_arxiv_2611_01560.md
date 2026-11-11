# Democratizing Robot-Use Agents（arXiv:2611.01560）

- **类型：** paper；最新核对版本 **v2（2026-11-03）**，首次发布 v1（2026-11-01）。
- **作者：** Jiarui Yuan、Yuxin Zuo、Xiaoyu Huang 等；清华大学、加州大学伯克利分校。
- **原文：** <https://arxiv.org/html/2611.01560v2> · [PDF](https://arxiv.org/pdf/2611.01560v2) · [版本历史](https://arxiv.org/abs/2611.01560)
- **项目：** <https://uni-robot-agent.com/> · **代码：** <https://github.com/Open-RUA/uni-robot-agent>
- **博客：** <https://openrua.ai/uni-robot-agent> · **文档：** <https://uni-robot-agent.com/docs/>
- **入库日期：** 2026-11-03；**复核日期：** 2026-11-11
- **核对范围：** 论文 v2 §1–8（问题、五层架构、调度/快照、仿真与真机评测、模型能力、消融、效率、相关工作及局限），附录 A–G（实验设置、真机 19 任务与逐项得分、成本估算、监控运动执行算法、几何代码案例、动作期间规划、rollout 数量敏感性）。对照作者博客与公开仓库，不将项目宣传当独立实验。
- **通读：** 已完成

## 核心证据（按原文分节）

1. **统一接口 (§2.1–2.2)：** Agent → orchestration → shared tools → services → native backends；环境适配器复用工具语义和反馈。交互数据为未来模型训练提供可组合记录，非论文已完成的模型训练结果。
2. **执行机制 (§2.3、§2.5)：** 后台 job handle、独占资源队列；观测快照绑定图像/相机/本体状态，请求去重，运动/夹爪协同，执行异常与误差回执，视觉历史滑动窗口。
3. **RoboDojo (§3.1、表 1)：** 42 任务，主要 3 seeds × 2 layouts = 252 episodes；Qwen3.8-Flash-Next + 本框架 **25.00% SR / 31.39 Score**；已发表 GPT-6 Astra + RoboProbe **22.48% / 28.97**；Astra + 本框架 **46.67% / 51.90**。这些跨模型/harness 分数并非控制变量实验；Score 含部分进度。
4. **真机 (§3.2、表 2、附录 B 表 4)：** 双臂 19 项，五维等权，Qwen + 本框架 **53.50/100**、Astra **80.50**（零样本且使用 D405 深度）；微调 π₀.₅ **47.75**、零样本 Dream-1 **43.50**（RGB）。各组观测与训练条件不同，分数不是成功率。v1 的旧 49.50 与 v2 表 2 / 表 4 和博客中的 **53.50** 已按最新版本统一。
5. **能力分析 (§4)：** 视觉理解、跨视角空间 grounding、几何计算代码、具身推理四类常见失败；模型代际/家族/规模对比具有相关分析性质。
6. **特权信息与技能 (§5)：** Qwen 去 depth 时 35.89 → 31.39，进一步去相机参数为 9.13；禁用技能库为 30.31 Score / 24.58% SR。
7. **成本/速度 (§6、附录 C)：** token、缓存及货币价格估算有日期和口径；五个单回合对照的动作-模型重叠 **84.5%–93.8%**，回合耗时降低 **4.8%–73.7%**，不能外推为统计显著加速。
8. **算法/边界（附录 D–G）：** 运动按实测位置生成 waypoint、监测发散/停滞/异常位姿跳变；目标 pose 到达不意味着任务完成；RGB 场景靠多视角几何代码估计空间；附录 G 的少 rollout 测试属于独立深度设置，不能当作主评测置信区间。

## 对 wiki 的映射

- [Uni-Robot-Agent 主节点](../../wiki/entities/paper-uni-robot-agent.md)
- [项目归档](../sites/uni-robot-agent.md)
- [源码归档](../repos/uni-robot-agent.md)
