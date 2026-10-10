# PSO 一手资料：奠基论文、惯性权重、收缩因子与标准实现

> 来源归档（ingest）；保存书目信息与归纳，不转存受版权保护的论文全文。

- **类型：** paper / foundational-algorithm / technical-report
- **入库与核查日期：** 2026-10-10
- **一句话说明：** 从 1995 年原始 PSO 到惯性权重、收缩因子及 SPSO 标准基线，区分历史提出、理论分析与实现版本。
- **配套资料：** [Particle Swarm Central](../sites/particle_swarm_central.md)、[SPSO-2011 C 参考源码](../repos/standard_pso_2011.md)

## 核心论文摘录

### 1）A New Optimizer Using Particle Swarm Theory（Eberhart / Kennedy，1995）

- **作者：** Russell C. Eberhart、James Kennedy。
- **会议：** MHS'95，1995-10-04 至 10-06，日本名古屋；pp. 39–43。
- **DOI / 出版社：** <https://doi.org/10.1109/MHS.1995.494215>；<https://ieeexplore.ieee.org/document/494215/>
- **核查层级：** 出版社元数据与摘要；本次未获取该篇完整正文，不据摘要补写实验数值。
- **贡献：** 比较粒子群求解非线性函数的两种范式，含局部导向版本；提出神经网络训练、机器人任务学习等潜在应用。
- **历史辨析：** 是 1995 年奠基工作之一；IEEE 的 2002 年收录时间不是论文提出时间。摘要里的机器人任务学习是应用提议，不是现代人形机器人部署证明。
- **对 wiki 的映射：** [粒子群优化](../../wiki/methods/particle-swarm-optimization.md) — 起源与拓扑；[AutoPSO](../../wiki/entities/paper-autopso.md) — 基础概念入口。

### 2）Particle Swarm Optimization（Kennedy / Eberhart，1995）

- **作者：** James Kennedy、Russell Eberhart。
- **会议：** ICNN'95，Perth，Vol. 4，pp. 1942–1948。
- **DOI / 出版社：** <https://doi.org/10.1109/ICNN.1995.488968>；<https://ieeexplore.ieee.org/document/488968/>
- **原文 PDF（大学站点镜像）：** <https://staff.washington.edu/paymana/swarm/kennedy95-ijcnn.pdf>
- **核查层级：** 论文原文；镜像承载原始研究，不把镜像提供者当作作者。
- **贡献：** 说明从社会行为模拟到连续优化的演化；粒子保存自身历史最好位置，并参考群体经验更新速度与位置。
- **与 GA 区别：** 标准 PSO 不以交叉、变异、繁殖替换个体为基本更新算子；群体数量与粒子记忆是另一种搜索机制。
- **对 wiki 的映射：** [粒子群优化](../../wiki/methods/particle-swarm-optimization.md) — 位置、速度、pbest 与社会项。

### 3）A Modified Particle Swarm Optimizer（Shi / Eberhart，1998）

- **作者：** Yuhui Shi、Russell Eberhart；论文署名单位为印第安纳大学–普渡大学印第安纳波利斯分校（IUPUI），不能套用作者今天的任职单位。
- **会议：** IEEE International Conference on Evolutionary Computation，pp. 69–73。
- **DOI / 出版社：** <https://doi.org/10.1109/ICEC.1998.699146>；<https://ieeexplore.ieee.org/document/699146/>
- **作者上传原文：** <https://www.researchgate.net/publication/3755900_A_Modified_Particle_Swarm_Optimizer>（页面明确标注 Yuhui Shi 上传；核查的是论文正文，不引用平台生成的相关文章摘要）。
- **贡献：** 为旧速度乘上惯性权重，控制保留原运动方向的程度；实验还讨论随迭代降低惯性。
- **边界：** 文中自述测试问题范围有限；惯性调度不是普适最佳参数，也不是 1995 年原版就有的完整配置。
- **对 wiki 的映射：** [粒子群优化](../../wiki/methods/particle-swarm-optimization.md) — 惯性权重方程与参数读法。

### 4）The Particle Swarm—Explosion, Stability, and Convergence in a Multidimensional Complex Space（Clerc / Kennedy，2002）

- **作者：** Maurice Clerc、James Kennedy。
- **期刊：** IEEE Transactions on Evolutionary Computation，6(1)，58–73，2002-02。
- **DOI / 出版社：** <https://doi.org/10.1109/4235.985692>；<https://ieeexplore.ieee.org/document/985692/>
- **原文 PDF 镜像：** <https://iasei.org/pkucil/docs/20190117151846014665.pdf>
- **核查层级：** 原文，尤其收缩因子与粒子动力学分析；不是只读论文标题就断言全局最优保证。
- **贡献：** 分析速度爆炸与粒子轨迹稳定性，用收缩系数控制更新动力学。
- **边界：** 粒子轨迹的稳定 / 收敛条件不等同于对任意非凸目标找到全局最优的证明。
- **对 wiki 的映射：** [粒子群优化](../../wiki/methods/particle-swarm-optimization.md) — 收缩式与惯性式区别；[数值优化方法选型](../../wiki/queries/numerical-optimization-method-selection.md) — 黑箱搜索的保证边界。

### 5）Standard Particle Swarm Optimisation（Maurice Clerc，2012）

- **作者报告：** <https://hal.science/hal-00764996>；HAL 提交 2012-12-13，正文版本 2012-09-23。
- **原文 PDF（大学站点镜像）：** <https://mat.uab.cat/~alseda/MasterOpt/SPSO_descriptions.pdf>
- **配套作者来源代码：** <https://www.particleswarm.info/standard_pso_2011_c.zip>，具体入口与 SHA-256 见[代码归档](../repos/standard_pso_2011.md)。
- **贡献：** 对照 SPSO-2006 / 2007 / 2011；SPSO-2011 改用几何式位置分布、随机信息拓扑，为比较改进算法提供固定基线。
- **边界：** SPSO 是研究参考标准，不是认证标准或“最强 PSO”；报告与具体 C 包的可选项、顺序更新要分别核对，不能拿任意三项更新脚本冒充 SPSO-2011。
- **对 wiki 的映射：** [粒子群优化](../../wiki/methods/particle-swarm-optimization.md) — 标准实现与复现检查；[CMA-ES](../../wiki/methods/cma-es.md) — 黑箱优化对照。

## 访问与开放状态

IEEE 页面本次有 JavaScript 验证限制；以公开索引元数据、原文镜像和明确的作者上传稿交叉核查。Clerc 个人站与 HAL 下载入口本次未能读取全文，报告使用上述公开大学镜像；Particle Swarm Central 的 C ZIP 已实际下载并检查。保留正规 DOI / HAL 入口，不将镜像访问失败描述为论文或代码从未公开。
