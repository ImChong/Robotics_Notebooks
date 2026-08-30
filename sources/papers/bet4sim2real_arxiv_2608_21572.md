# Bet4Sim2Real（仿真下注性能证书）

> 来源归档（ingest）

- **标题：** Betting for Sim-to-Real Performance Certificates
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.21572>
- **机构：** 艾奥瓦州立大学（Iowa State University）
- **作者：** Yujia Chen、Bowen Weng
- **代码：** <https://github.com/ISUSAIL/Bet4Sim2Real-Certificate>
- **入库日期：** 2026-08-30
- **一句话说明：** 在每次真机结果揭晓前参考大规模模拟器库下注，把累计财富转成 anytime-valid 的性能区间，减少昂贵真机试验。

## 核心摘录（MVP）

### 1) 小样本证书过宽

- **摘录要点：** 真机试验贵，经典均值置信区间往往过宽。作者把仿真库变成逐次赌注：真实结果结算财富并调整对各模拟器的信任，再把财富映射为性能区间。
- **对 wiki 的映射：**
  - [Bet4Sim2Real](../../wiki/entities/paper-bet4sim2real.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 2) anytime-valid 与财富遗憾

- **摘录要点：** 返回证书对任意模拟器库保持 anytime-valid（任意停止时刻覆盖真实均值）。财富遗憾界指导算法与模拟器库配置，使证书收紧。
- **对 wiki 的映射：**
  - [Bet4Sim2Real](../../wiki/entities/paper-bet4sim2real.md)

### 3) 评测数字

- **摘录要点：** 合成分布 + 真机回放 / 在线评测。相对经典及先进基线，证书平均收窄 **51.6%±16%**；≤30 样本仍收窄 **32.26%±8%**。仓内复现：GR00T G1 命令跟踪、NIST 连续移动操作、ASTM WK86916 Go2 推倒。
- **对 wiki 的映射：**
  - [bet4sim2real 仓库](../repos/bet4sim2real.md)

### 4) 开源状态（截至 2026-08-30）

- **摘录要点：** **已开源**（README 未写 SPDX）。`synthetic/method` 为证书实现；三组机器人研究从该目录导入。含 `supplement.pdf`。

## 当前提炼状态

- [x] arXiv 摘要与仓库目录对齐
- [x] wiki 映射：`wiki/entities/paper-bet4sim2real.md` 新建
