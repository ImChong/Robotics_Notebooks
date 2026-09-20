# 常微分方程如何选择求解方法：从解析法的决策链到数值法的误差与稳定性

> 来源归档（blog / 微信公众号）

- **标题：** 常微分方程如何选择求解方法：从解析法的决策链到数值法的误差与稳定性
- **类型：** blog
- **作者：** 待核实（公众号账号 WebFetch 未解析出 nick_name）
- **原始链接：** https://mp.weixin.qq.com/s/lwU-BERDRRKifsF7nr8wkg
- **入库日期：** 2026-09-20
- **抓取方式：** WebFetch 直拉 `mp.weixin.qq.com` 正文（本环境无预装 `wechat-article-for-ai`）
- **原始抓取落盘：** [`sources/raw/wechat_ode_solving_methods_decision_chain_2026-09-20.md`](../raw/wechat_ode_solving_methods_decision_chain_2026-09-20.md)
- **一句话说明：** 从阶数/线性/存在唯一性出发，给出 ODE 解析求解决策链（可分离、线性、恰当、齐次型、Bernoulli、降阶、常系数、Laplace、方程组）与变量替换统一视角；数值部分覆盖 Euler/RK4/隐式/A-stable、自适应步长、多步法、二阶专用格式（辛 Euler、Störmer–Verlet、Newmark-β、Numerov）及检验要点。
- **步骤 2.5（开源核查）：** 科普教程文，无单一项目页；引用的 MATLAB `ode45`、SciPy `solve_ivp` 等为公开库接口。
- **沉淀到 wiki：** [`wiki/formalizations/ode-solving-methods.md`](../../wiki/formalizations/ode-solving-methods.md)

## 核心摘录（归纳，非全文）

### 解析：先诊断再选路

1. **问题检查：** 阶数 → 线性 → 标准形 → Peano 存在 / Lipschitz 唯一
2. **一阶链：** 直接积分 → 可分离 → 一阶线性（积分因子）→ 恰当 → 齐次型（\(y/x\)）→ Bernoulli → 组合代换 → 数值/定性
3. **高阶链：** 降阶（不显含 \(y\) 或 \(x\)）→ 常系数（特征根）→ 非齐次（待定系数 / 常数变易 / Laplace）→ Euler–Cauchy → 线性方程组 → 数值
4. **解后四检：** 代回、除法丢解、定义域与独立常数个数、初值存在唯一性

### 变量替换统一原则

- 由**对称性/不变性**提出候选变量 → 链式法则求导 → **闭合**（新方程只含新变量）→ **可逆**（隐函数定理 / 分区间）
- 边界：改自变量、降阶、Laplace 变换不在「只换因变量」框架内

### 数值：精度 vs 稳定性 vs 结构

| 主题 | 要点 |
|------|------|
| 显式 vs 隐式 | 未知量是否出现在右端；隐式每步解代数方程 |
| Euler | 局部 \(O(h^2)\)，全局 \(O(h)\)；步长过大可不稳定 |
| 改进 Euler / RK4 | 二阶 / 四阶；RK4 权重由 Taylor 四阶匹配（附录） |
| 后向 Euler / 梯形 | A-稳定；后向 Euler 亦 L-稳定；梯形非 L-稳定 |
| 自适应 | 嵌入 RK（Dormand–Prince RK45）；FSAL；`ode45` / `solve_ivp` |
| 多步法 | Adams–Bashforth（显式）/ Adams–Moulton（隐式）；BDF 适合刚性 |
| 二阶专用 | 一般问题降一阶系统；保守系统用 Störmer–Verlet；结构动力学 Newmark-β；Numerov 用于特殊二阶形 |
| 检验 | 收敛阶、稳定性/刚性、守恒、基准解、问题可解性 |

### 三类数值困难

- **普通非刚性：** 精度主导步长
- **刚性：** 快模态限制显式步长 → 隐式 / BDF
- **保守长期积分：** 关注能量漂移 → 辛格式

## 对 wiki 的映射

- [ode-solving-methods](../../wiki/formalizations/ode-solving-methods.md)（本次升格主页面）
- [damped-systems](../../wiki/formalizations/damped-systems.md)（二阶响应与特征根读法）
- [eigenvalues-eigenvectors](../../wiki/formalizations/eigenvalues-eigenvectors.md)（常系数特征方程）
- [lyapunov](../../wiki/formalizations/lyapunov.md)（存在唯一与稳定性语言）
- [robot-simulation-three-layers](../../wiki/concepts/robot-simulation-three-layers.md)（仿真积分器选型语境）
