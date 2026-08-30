# Bet4Sim2Real-Certificate（ISUSAIL）

> 来源归档

- **标题：** Betting for Sim-to-Real Performance Certificate
- **类型：** repo
- **来源：** 艾奥瓦州立大学（Iowa State University）
- **链接：** <https://github.com/ISUSAIL/Bet4Sim2Real-Certificate>
- **论文：** <https://arxiv.org/abs/2608.21572>
- **许可：** README 未声明 SPDX
- **入库日期：** 2026-08-30
- **一句话说明：** 仿真下注证书的可复现实现：合成实验 + GR00T G1 / NIST / ASTM Go2 三组回放。
- **沉淀到 wiki：** [`wiki/entities/paper-bet4sim2real.md`](../../wiki/entities/paper-bet4sim2real.md)

---

## 仓库入口（README）

| 目录 | 复现 |
|------|------|
| `synthetic/` | Fig. 1–4；证书实现在 `synthetic/method` |
| `gr00t_command_tracking/` | Fig. 5；Unitree G1 + GR00T 命令跟踪，匹配 MuJoCo 模拟器库 |
| `nist_continuous_manipulation/` | Fig. 6a；NIST 连续移动操作 peg-in-hole 回放 |
| `astm_wk86916_go2/` | Fig. 6b；ASTM WK86916 Go2 推倒回放 |
| `supplement.pdf` | 命题与附录证明 |

各子目录自带 README、数据说明与配置。

## 开源边界（截至 2026-08-30）

- **已开源**：证书算法与论文图复现脚本可辨识。
- **许可：** 根目录无 LICENSE。
- **真机：** 以标准化测试回放为主，不是在线训练策略。
