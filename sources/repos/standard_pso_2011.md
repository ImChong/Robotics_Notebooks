# Standard PSO 2011（Particle Swarm Central C 参考源码）

- **类型：** source-code / reference-implementation（无 GitHub 仓库，不虚构仓库身份）
- **作者来源：** Maurice Clerc / Particle Swarm Central；包内 `main.c` 标明来源并给出 Clerc 联系信息。
- **项目页：** <https://www.particleswarm.info/Programs.html>
- **代码：** <https://www.particleswarm.info/standard_pso_2011_c.zip>
- **报告：** <https://hal.science/hal-00764996>；[论文归档](../papers/pso_foundations_1995_2012.md)
- **核查日期：** 2026-10-10
- **快照 SHA-256：** `11692f658158b18aafd97d667eeebdc7527cf21147d530d73d4c7eb795af0557`
- **开放状态：** 源码公开可下载；ZIP 无明确独立许可证文件，不能推定商业再分发许可。
- **一句话说明：** 实际核查的历史 C 包，用几何式更新与随机信息链接实现 SPSO-2011，另含可选实验机制。

## 实际文件与运行入口

| 文件 | 核查用途 |
|---|---|
| `ReadMe.txt` | 标准动机、更新记录、可选 bells and whistles，并声明不是市场最优算法 |
| `main.c` / `main.h` | 入口、参数、问题编号、重复运行与输出；`main.c` 末尾直接 include 多个 `.c` 文件 |
| `PSO.c` | 初始化 X / V / P、信息链接、邻域最优、几何更新、边界修复、评价预算 |
| `problemDef.c` / `perf.c` | 搜索空间、目标与测试函数评分 |
| `alea.c` / `KISS.c` / `mersenne.c` | 随机数与准随机数支持 |
| `f_run.txt` / `f_synth.txt` / `f_trace.txt` | 程序输出文件；运行会以写模式打开，不应覆盖自己保留的实验记录 |

## 源码核查要点

- 包内默认群体大小 40、`w=1/(2 ln 2)`、`c=0.5+ln 2`；不是收缩式常见的 `chi≈0.7298 / phi1=phi2=2.05` 那套方程。
- `PSO.c` 按粒子顺序评价并立即更新记忆；同轮后续粒子可看见前面粒子的改进。`BW[3]` 控制顺序置乱，下载包默认为 0；不要仅凭报告概述推断包内总是随机顺序或同步更新。
- 默认拓扑用随机链接矩阵，不是全局 gbest 广播；没有全局改进时重新构造链接。
- `main.h` 包含 GSL 准随机接口；即便运行未选择准随机选项，编译仍需 GSL 头文件与链接库。只编译 `main.c`，不可再把被 include 的 `.c` 重复编译链接。
- `ReadMe.txt` / 代码的非标准可选项要记入实验配置；其结果不能不加说明地冠名标准基线。
- 本次验证 ZIP 获取、哈希与入口 / 函数，未编译运行此历史程序，未复现包内 `Results.pdf` 数值。原始源码 / PDF 不随本次 PR 转存。

## 对 wiki 的映射

- [粒子群优化](../../wiki/methods/particle-swarm-optimization.md) — 源码运行时序、SPSO 与教学方程的区别。
- 配套：[资料站](../sites/particle_swarm_central.md)、[一手论文](../papers/pso_foundations_1995_2012.md)。
