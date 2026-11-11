# GRASP（arXiv:2611.05754）

> 来源归档（paper）

- **标题：** GRASP: Gradient-Guided Sampling for Contact-Rich Physics-Based Motion Retargeting
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.05754>
- **PDF：** <https://arxiv.org/pdf/2611.05754>
- **项目页：** <https://computationalrobotics.seas.harvard.edu/GRASP/>
- **代码：** <https://github.com/ComputationalRobotics/GRASP_code>
- **入库日期：** 2026-11-05
- **线索来源：** [人能稳稳提起的罐子，换一双机器手怎么提？](../blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)（具身智能小站，2026-11-05）
- **一句话说明：** 人的手指能扣住提手，换一双指长和关节都不同的机器手，同样的手势却可能让容器滑落。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-05）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 换一双机器手，怎样复现提罐和倾倒？。GRASP 在物理仿真里同时跟踪手与物体，利用试过的控制轨迹来调整手指的落点和握持姿势，让机器手也能提起容器、稳稳倾倒。
- 人的手指能扣住提手，换一双指长和关节都不同的机器手，同样的手势却可能让容器滑落。
- 指标：公众号转述：真机灵巧手展示了开环提拿和倾倒，整批成功率来自仿真。

## 对 wiki 的映射

- [paper-grasp](../../wiki/entities/paper-grasp.md)
- [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-05-43-papers-technology-map.md)


- **核对范围：** 通读 arXiv:2611.05754v1 官方 HTML 全文（第 1–8 节、算法 1/2、实验表 1–4、附录第 9–14 节及消融表 5–9），交叉核验 Harvard 项目页和 GRASP_code 核心运行路径；未进行 GPU 实验复现。
- **通读：** 已完成

## 一手资料补充核验（2026-11-11）

- 作者：Yang Liu、Tong Wang（共同一作）、Yiming Bao、Heng Yang；机构：Harvard University, School of Engineering and Applied Sciences。
- arXiv 首版：2026-11-03；论文共 26 页（含参考文献和附录）。
- **已解决指标口径**：论文正文表 1 使用 **1,024 粒子**，在 OakInk2 的 1,352 条 Sharpa 双手示范上成功 1,254 条，即 **92.8%**；论文附录表 9 使用 **2,000 粒子**，成功率为 **94.16%**，项目页和论文表 2 将其四舍五入为 **94.2%**。不是同一粒子预算下的相互矛盾结果。跨本体同为 2,000 粒子：Sharpa 94.2%、Wuji 92.4%、XHand 95.0%。附录表 9 的误差统计还限定在三种粒子预算下均有有限数值的 1,305 条任务，而成功率分母仍为 1,352；比较时注意两种分母口径。
- 官方项目页定义成功条件：物体平均位置误差小于 10 cm，平均旋转误差小于 0.5 rad；执行失败计入失败。
- 项目主页报告的 Sharpa 对照：Dex-retargeting 28.6%、SPIDER mjwp 71.4%、SPIDER mjwp_act 77.1%、Do-as-I-Do 81.0%、GRASP 94.2%；项目页提示各基线评估集不完全一致，不能视为严格同协议横向对比。
- 核心算法：GRASP-TO 在 tempered SMC 中用历史 rollout 的控制—代价对拟合局部二次代理模型，恢复梯度引导 MCMC rejuvenation；用真实物理仿真代价与正反提议密度进行 Metropolis–Hastings 接受率校正。GRASP-MPC 采用 warm-start 的滚动时域规划扩展到长示范。
- 不需要可微分模拟器，也不需要任务专用接触奖励；优化参考轨迹用于 PPO 转笔和人形与物体交互跟踪；作者展示真机灵巧手和人形运动，但不能把仿真成功率解释为真机成功率。
- 源码地址：https://github.com/ComputationalRobotics/GRASP_code （与项目展示网站仓库 https://github.com/ComputationalRobotics/GRASP 区分）。
- 论文方法与附录：参考控制采用相邻 knots 线性插值及 AR(1) 高斯先验；GRASP-TO 在每个温度阶段按 ESS 自适应增温、重采样，再用局部二次拟合恢复梯度，引导 MALA 提议，并按真实仿真代价的 MH 比率接受/拒绝。GRASP-MPC 每窗口重用未执行计划与粒子库，最终重评候选并执行前缀。Gaussian smoothing 的梯度解释来自总体极限与附加假设，**不表示有限样本拟合梯度精确**。
- 更多原文结果：HOT3D 全量 5,400 条的 ADD-AUC 为 81.54%；九段转笔 PPO 最后 500 次迭代均值奖励从运动学参考 21.7 提升为 GRASP 参考 45.9，同时轴角误差由 3.9° 增至 4.3°；人形 G1 的 LAFAN1、AMASS、OMOMO 主要跟踪指标详见论文表 4。
- 代码入口已核对：`experiments/paper.py:main/build` 组装参数并直接调用 `grasp.mpc.run:main`；后者通过 `grasp.utils.io.load_data`、`grasp.sim.mjwp.setup_env`、`grasp.sim.rollout.make_rollout_fn` 构造 `GraspWindowEvaluator` 与 `GraspMPC`；`GraspMPC.run` 调用 `GraspSampler.run`，后者利用 `grasp.optim.surrogate.fit_quadratic_surrogate` 及 `sampler._propose/_log_alpha` 执行采样/MH 校正；最终 `grasp.mpc.run` 复评、计算指标并保存 NPZ/JSON。仅静态阅读核心执行路径，**未审计全部代码文件，也未运行 GPU / 真实机器人实验**。

### 论文方法流程（依据官方项目页）

```mermaid
flowchart TD
    A["人体运动 / 物体参考轨迹"] --> B["机器人控制候选与物理仿真"]
    B --> C["记录控制—真实代价对（接受与拒绝样本）"]
    C --> D["局部二次代价模型拟合"]
    D --> E["恢复梯度，生成引导式 MCMC 提议"]
    E --> F["物理仿真计算真实代价"]
    F --> G["Metropolis–Hastings 校正"]
    G --> C
    G --> H["GRASP-TO 优化轨迹"]
    H --> I["GRASP-MPC 滚动时域 warm-start"]
    I --> J["可用于 PPO 跟踪的物理参考轨迹"]
```

### 一手来源

1. [arXiv:2611.05754v1（官方全文 HTML，含附录）](https://arxiv.org/html/2611.05754)
2. [Harvard GRASP 官方项目主页（方法、评估表格、成功定义）](https://computationalrobotics.seas.harvard.edu/GRASP/)
3. [官方代码仓库](https://github.com/ComputationalRobotics/GRASP_code)
