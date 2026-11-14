# 【MPC】四足机器人感知型 NMPC 的高程图凸分解方法

> 来源归档：微信公众号技术解读 / 用户提供 PDF（非原始论文）

- **标题：** 【MPC】四足机器人感知型 NMPC 的高程图凸分解方法
- **作者 / 发布：** Spgroc / 古月居（网页署名）
- **发布时间：** 2026-11-14（用户提供 PDF 页眉）
- **类型：** blog
- **链接：** https://mp.weixin.qq.com/s/0GIh1h4aBKJn23UHESaxGQ
- **附件：** 用户提供的 19 页网页导出 PDF《【MPC】四足机器人感知型NMPC的高程图凸分解方法.pdf》；本仓库只归档提炼，不转载 PDF 正文与图片
- **核对范围：** 通读 PDF 第 1–14 页正文（问题定义、质心 NMPC、高程图、凸分解、距离场、参考变形、频率预算、仿真说明、结论及文献）；第 15–19 页为空白/阅读原文尾页。PDF 的部分公式和表格在导出中未显示，未据此补造参数
- **通读：** 已完成
- **入库日期：** 2026-11-14
- **性质：** 对算法链路和 Go2/MuJoCo 仿真案例的二次解读；不是 Grandia 等人的原始论文，也不是独立可复现实验报告
- **项目源码线索：** [Khaledsweid1/ocs2-perceptive-mpc-go2](https://github.com/Khaledsweid1/ocs2-perceptive-mpc-go2)；该仓 README、采样尺寸、控制栈和仿真场景与解读吻合，但**附件并未直接列出仓库 URL，不能确认其就是文章唯一来源**
- **映射：** [Go2 感知型 OCS2 NMPC 适配项目](../../wiki/entities/ocs2-perceptive-mpc-go2.md)

## 原文技术主线（提炼）

1. 地形高度场 → 局部高程栅格：地图中心随机器人移动，文章例子为 6.0 m × 6.0 m、0.04 m/格；特别提醒 `grid_map` 索引方向与机器人自身几何体误采样问题。
2. 高程栅格 → 局部平面估计与可落足分类 → 连通踏面 → 凸多边形 / 半平面落足可行域；梯度陡变处不应被当成可稳定支撑的踏面。
3. 地形信息另外形成摆动足 / 腿部的符号距离净空约束，以及基座高度、滚转、俯仰参考，进入接触模式切换的 NMPC。
4. 有限时域滚动优化由 WBC/关节层跟踪：文章报告硬件 1000 Hz、WBC 500 Hz、地形 5 Hz、指令 20 Hz，NMPC 异步；单次 WBC 约 0.23 ms、NMPC 平均 7.2 ms/峰值 12.1 ms、凸分解约 3.6 ms。
5. Unitree Go2 + MuJoCo + ROS 2 为文章所述仿真背景；插图中的部分曲线用于算法解释，文章明确**不代表新物理实验数据**（第 11 页）。

## 与一手资料的核对与差别

- **原创方法归属：** [Grandia 等，Perceptive Locomotion Through Nonlinear Model-Predictive Control，IEEE T-RO 2023](https://doi.org/10.1109/TRO.2023.3275384)（[arXiv 全文](https://arxiv.org/abs/2208.08373)）研究的是 ANYmal 上的感知型 NMPC；论文实验不能转述成 Go2 真机验证。
- **Go2 开源适配：** [Khaledsweid1 仓库](https://github.com/Khaledsweid1/ocs2-perceptive-mpc-go2)公开的是 ROS 2 Jazzy + MuJoCo + OCS2 上游补丁与启动工具，非原创控制算法。
- **重要实现差异：** 适配项目 `tools/launch/perceptive_mujoco.launch.py` 的 `fp` 缺省为 `0`；其 `docs/PERCEPTIVE_MPC_GUIDE.md` 明确落足区域约束默认关闭，启用后可能过约束，整段楼梯尚未爬完。因此不要把“计算出了绿色凸区域”直接等同于“优化器默认强制落足其中”。
- **传感器边界：** 本实现的 `terrain_elevation_pub.py` 从 MuJoCo 静态几何体射线采样，**不是**真实相机/雷达感知。未命中/命中机器人时具体实现写为高度 0.0，并非完全实现了文章叙述的“无效格剔除”。

## 为什么收录

把非光滑地形栅格转成 MPC 可处理的局部几何约束，是连接地形感知、动力学规划、实时控制的典型范式；对工程复现尤其要区分论文方法、ROS 2 适配层、仿真可视化与实际开启的约束。

## 原始出处

- [古月居文章（附件来源）](https://mp.weixin.qq.com/s/0GIh1h4aBKJn23UHESaxGQ)
- [Grandia 等原论文（2023，arXiv:2208.08373）](https://arxiv.org/abs/2208.08373)
- [Go2 适配源码与文档](https://github.com/Khaledsweid1/ocs2-perceptive-mpc-go2)
