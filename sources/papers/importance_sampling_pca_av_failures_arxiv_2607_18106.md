# Importance Sampling and PCA for Finding Failures in Commercial Autonomous Vehicles（arXiv:2607.18106）

> 来源归档（ingest）

- **标题：** Importance Sampling and PCA for Finding Failures in Commercial Autonomous Vehicles
- **类型：** paper / autonomous-driving / safety-validation / importance-sampling / rare-event / PCA
- **arXiv：** <https://arxiv.org/abs/2607.18106>（v1，2026-07-20；PDF：<https://arxiv.org/pdf/2607.18106>）
- **会议：** IEEE ICVES 2026（Submitted）
- **作者：** Hailey Warner*、Duncan Eddy、Shreya Parjan、Caroline Cahilly、Harrison Delecki、Matthias Kleinstäuber、Chaitanya Shinde、Jerry Lopez、Mykel J. Kochenderfer
- **机构：** 斯坦福大学航空航天系（Stanford AA；*通讯）；Torc Robotics（Blacksburg, VA）
- **资助：** Torc Robotics via Stanford Center for AI Safety；Stanford Graduate Fellowship；Stanford HAI
- **入库日期：** 2026-08-01
- **一句话说明：** 把 AST（SAC 对抗搜失败噪声轨迹）与 DiFS（扩散采样多样失败）首次接到 **商业自动驾驶卡车规划栈**（Object Sim + ROS 黑盒）；Monte Carlo 找不到碰撞时二者可找到；再用 PCA + K-means 抽出可复现的「eigenfailures」做感知级诊断。

## 开源状态（步骤 2.5）

- **项目页：** 论文 / abs **未列**独立项目页或 GitHub。
- **代码 / 数据：** 无公开训练/评测仓库；实验对象为 **商业卡车规划器**（规则层次路径采样 + QP 安全舒适优化 + 安全覆盖；经 ROS 接 Applied Intuition Object Sim；AST/DiFS 经 ZeroMQ 黑盒注入最近车辆位置噪声）。
- **结论（截至 2026-08-01）：** **确认未开源**。方法可复述，但商业栈与仿真耦合不可社区复现；复现需自备可注入观测噪声的规划器 + 仿真接口。

## 摘录 1：问题与主张（§I）

- **痛点：** 学习式 importance sampling（AST、DiFS）此前多在 IDM 等学术/开源简单驾驶模型上验证；商业规划器更鲁棒、失败更稀，能否迁移未知。原始噪声轨迹 alone 也难变成可行动诊断。
- **主张：**（1）首次把 AST 与 DiFS 接到商业 AV 卡车栈；（2）对失败噪声轨迹做 PCA → 聚类 → 反变换，得到可复现的典型感知噪声模式（eigenfailures）。
- **场景：** 高速 cut-in / merge；对最近 actor 的感知位置加性噪声（纵向/横向）；碰撞即失败。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-importance-sampling-pca-av-failures.md`](../../wiki/entities/paper-importance-sampling-pca-av-failures.md)；与 [Safe RL](../../wiki/methods/safe-rl.md)、[SAC](../../wiki/methods/sac.md)、[Safety Filter](../../wiki/concepts/safety-filter.md)、[扩散模型](../../wiki/concepts/diffusion-model.md)、[自动驾驶核心算法地图](../../wiki/overview/autonomous-driving-core-algorithms-series.md) 互链。

## 摘录 2：方法栈（§II）

- **黑盒设定：** 栈收观测出控制；每步对最近车辆感知位置注入扰动 \(a_t\)；噪声轨迹 \(x=(a_1,\ldots,a_T)\)；碰撞 → fail。
- **目标分布：** \(p^\star(x\mid \mathrm{fail}) \propto \mathbf{1}[\mathrm{collision}]\,p(x)\)。\(p(x)\) 取零均值高斯，\(\sigma\) 随距离线性增大（作感知规格上界，而非标定真实传感器）：\(\sigma_x=0.02x+1\)，\(\sigma_y=0.00625y+0.2\)。
- **AST：** 把注入噪声建成 MDP；奖励为逐步 \(\log p(a)\)，非失败终止罚 \(\alpha + c_{\mathrm{dist}} d_{\min}\)；用 **SAC** 学对抗策略（Fig.2）。
- **DiFS：** 迭代「从当前扩散模型采样 → 按鲁棒性 \(r=\min\) 距离过滤最差分位 → 再训」；擅长多样失败（Fig.3；阈值 0.3）。
- **PCA 诊断：** 对失败噪声集 \(D\) 做 SVD/PCA → 主分量按时步指示敏感时刻 → K-means 聚类 → 反变换 \(\hat D = D^\ast V_k^\top + \mu\) 得广义噪声轨迹（eigenfailures）。

**对 wiki 的映射：** 实体页画「噪声先验 → AST/DiFS → PCA → eigenfailure 回放」流程图；强调与 Safety Filter / Safe RL 的互补（发现 vs 约束）。

## 摘录 3：评测要点（§III–§IV / Table I–VI）

| 对比 | 要点 |
|------|------|
| MC：商业 vs IDM | 商业失败率 **0.0%**、平均严重度 0；IDM **40.1%** / 0.719（Table I） |
| AST 超参（300 ep 扫） | 小 batch（≤16）、较大 buffer、小 \(\tau\) 抬高 log-prob；\(\alpha\)/\(c_{\mathrm{dist}}\) 权衡失败率 vs 似然；主实验选 \(\alpha{=}5000\)、\(c_{\mathrm{dist}}{=}1000\)、\(\tau{=}0.001\)、lr \(3\!\times\!10^{-4}\)、batch 16、buffer \(10^4\)（Table II） |
| 训练（1000 ep） | AST 失败率 **87.7%**、约 \$0.012/碰撞（T4）；DiFS **2.4%**、约 \$0.648/碰撞，但平均 log-prob 更高（−1496 vs −4506）（Table III） |
| 评测 | MC **2000** ep **0** 失败；AST **300** ep **94.6%**；DiFS **300** ep **3.1%** 但似然最好（Table IV） |
| 泛化 | 预训练 AST 策略在 cut-in ±5 m 上 **100/100** 仍碰撞（Table V） |
| 代理安全指标（DiFS 300） | 9 碰撞 + 10 near-miss；MinTTC &lt; 1.5 s / DRAC &gt; 7 m/s²；4 不可避、5 判为规划器失败（Table VI） |
| PCA | 横向噪声低方差线性结构；纵向影响更大；3 簇 eigenfailures 回放到相同/相近场景可复现失败；AST 有模式坍缩与碰撞前不必要的反向偏置 |

**对 wiki 的映射：** 用「MC 失效 → AST 样本效率 / DiFS 多样性 → PCA 可行动诊断」写选型读法；写明未开源与 sim-to-real 未验证。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-importance-sampling-pca-av-failures.md`**（含流程总览；源码运行时序图标不适用）。
- 注册机构 **`torc`**（Torc Robotics）；`stanford` 已有。
- 交叉：Safe RL、SAC、Safety Filter、扩散模型、自动驾驶核心算法地图；必要时轻触 sim2real（文中声明 gap 未探）。
