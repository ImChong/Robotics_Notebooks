# spider_scalable_physics_informed_dexterous_retargeting

> 来源归档（ingest）

- **标题：** SPIDER: Scalable Physics-Informed Dexterous Retargeting
- **类型：** paper
- **来源：** arXiv abs / arXiv HTML / 项目页（GitHub Pages）
- **原始链接：**
  - <https://arxiv.org/abs/2511.09484>
  - <https://arxiv.org/html/2511.09484v2>
  - <https://jc-bao.github.io/spider-project/>
- **作者：** Chaoyi Pan, Changhao Wang, Haozhi Qi, Zixi Liu, Homanga Bharadhwaj, Akash Sharma, Tingfan Wu, Guanya Shi, Jitendra Malik, Francois Hogan（FAIR at Meta；Carnegie Mellon University）
- **入库日期：** 2026-05-17；2026-11-07（全文通读更新）
- **核对范围：** 通读 arXiv:2511.09484 v3 正文 §1–§5（摘要、引言、相关工作、问题式 1a、采样优化与退火核算法 1、鲁棒化 / 数据增广 / 虚拟接触引导、灵巧手与人形的表 1–2、策略训练、真机、数据集生成、局限）；附录 A–C 的实现细节与逐任务结果表仅浏览，未逐项核对。未运行官方代码。
- **通读：** 已完成
- **一句话说明：** 面向**仅有运动学**的人体演示（动捕 / 视频 / VR 等），用**并行物理仿真中的采样式轨迹优化**把「人机运动学对齐 + 物体运动」 refinement 成**动力学可行、接触序列正确**的机器人轨迹；引入**课程式虚拟接触引导**降低接触丰富任务中的解歧义难度，并报告**跨 9 种人形/灵巧手、6 套数据**的规模化数据生成与下游 RL 加速。

## 核心摘录

### 1) SPIDER（Pan 等，arXiv:2511.09484，2025/2026）
- **链接：** <https://arxiv.org/abs/2511.09484>；HTML：<https://arxiv.org/html/2511.09484v2>；项目页：<https://jc-bao.github.io/spider-project/>
- **问题：** 灵巧与人形策略需要大规模演示，但**机器人本体数据采集**昂贵；人体运动数据丰富，却存在**具身差异**且通常**缺少力/力矩与真实接触**信息，运动学映射结果往往**动力学不可行**或接触意图丢失。
- **核心分工（论文叙事）：** 人体演示提供**全局任务结构与目标**；**大规模物理仿真采样**（退火核的采样型优化，类 MPC/CEM 思路）在仿真里把轨迹 refinement 到**动力学可行**且**接触序列合理**；不把问题写成「每条轨迹单独训一个 RL 策略」。
- **虚拟接触引导：** 在采样早期于机器人与物体之间加入**虚拟力**使物体**贴合期望接触点**，再随优化进程**逐步放松**，用课程式方式减少接触歧义、提升成功率（论文报告相对无该引导的采样基线约 **+18%** 成功率量级，以原文实验为准）。
- **规模化与效率主张：** 在多种灵巧手与人形平台上扩展；相对论文中的 **RL 重定向基线**声称约 **10×** 更快；生成约 **2.4M 帧**级别的动力学可行机器人数据用于策略学习（另有 **262 episodes / 800 小时** 等量纲叙事，细节以论文统计为准）。
- **下游：** 轨迹可直接部署（配合 domain randomization 等鲁棒化）、单演示**物理环境/物体增强**、以及加速 **RL** 等闭环学习。
- **代码/站点：** 论文配套站点与可视化见 <https://jc-bao.github.io/spider-project/>；网站源码仓 <https://github.com/jc-bao/spider-project>（页面自述为 project website；完整数据生成管线发布情况以论文与仓库更新为准）。
- **对 wiki 的映射：**
  - 新建 [SPIDER（物理感知采样式灵巧重定向）](../../wiki/methods/spider-physics-informed-dexterous-retargeting.md)
  - 在 [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)、[GMR](../../wiki/methods/motion-retargeting-gmr.md) 中补充「运动学参考 + 并行仿真采样 + 接触课程」的定位与交叉引用

## 当前提炼状态

- [x] 摘要 + 引言贡献点 + 方法主线（采样优化、虚拟接触、鲁棒化段落标题）已摘录
- [x] wiki 方法页与流程图已落盘
- [x] 与 motion-retargeting / pipeline / GMR 交叉引用已补

## 全文核对要点（2026-11-07，均出自论文）

- **问题式（1a）：** $\min_{u_{0:T-1}}\|x_T\ominus x_T^{ref}\|_{Q_T}^2+\sum_t(\|x_{t+1}\ominus x_{t+1}^{ref}\|_{Q_t}^2+\|u_t\|_{R_t}^2)$，s.t. 仿真器动力学、状态 / 控制可行集；$\ominus$ 在 SE(3) 上用对数映射。$Q,R$ 取对角。
- **采样优化（§3.2，算法 1）：** MPPI / CEM 风格的加权平均更新 $U^{i+1}=U^i+\sum_j \exp(-J_j/\lambda)W_j/\sum_j\exp(-J_j/\lambda)$，噪声协方差随迭代 $i$ 与时域位置 $h$ 退火 $\Sigma_h^i=\exp(-\tfrac{N-i}{\beta_1N}-\tfrac{H-h}{\beta_2H})I$；每个样本用并行仿真 rollout 评分；improvement 小于 $\epsilon_{tol}$ 提前停止。**无梯度、无策略网络。**
- **物理增广（§3.3）：** 最坏情形鲁棒化 $J_{rob}=\max_{d\in D}J(U,d)$（摩擦 / 接触裕度 / 物体质量等）；换物体网格、地形、施加外力（演示 120 N / 240 N）；**虚拟接触引导**：对意图接触点对在物体-手指相对位置上加罚项，强度 $\eta_i$ 随迭代放松到 $\infty$（即课程式先"粘住"后放开），要求接触指示 $c_{k,t}$（可由仿真碰撞检测自动得到）。
- **结果：** 9 种人形 / 灵巧手本体、6 个数据集、3 种仿真器。灵巧手（表 1）：相对 RL 基线 ManipTrans / DexMachina，ADD-AUC 相当或更好（如 Oakink-Inspire 22.3 vs 19.4）且吞吐高一个数量级（2.5 vs 0.1 FPS）；相对运动学基线 ADD-AUC 翻倍以上，加接触引导（SPIDER+C）更高。人形（表 2）：开环跟踪误差低于 GMR（如 LAFAN1 关节误差 0.58° vs 1.08°），速度略降（23.1 vs 35.2 FPS）。策略训练：用 SPIDER 参考的杯子抓取策略任务进度 0.99±0.01，运动学参考仅 0.07±0.11；人形搬箱子收敛更快。数据集：摘要称 6,520 条、正文称 6,532 条（原文前后不一致），4 种手、191 个物体，完成 rollout 的 ADD-AUC 为 61.48%。
- **局限（§4.5）：** 感知误差（HAMER / FoundationPose / Trellis 重建）会导致语义错误的接触模式；严重遮挡；**开环轨迹仅在对齐的初始状态附近有效**，闭环策略蒸馏留待未来；无接触引导时可能收敛到另一种合法抓取。
