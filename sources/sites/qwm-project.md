# QWM 项目页

> 来源归档（site）

- **标题：** QWM: Q-Learning with World Models
- **类型：** site
- **链接：** https://pd-perry.github.io/qwm/
- **arXiv：** <https://arxiv.org/abs/2608.17163>
- **入库日期：** 2026-09-21
- **一句话说明：** 用世界模型在 Q-learning 上做测试时树搜索，策略与价值仅在海量真实转移上训练；Robomimic 与 LIBERO 上显著优于强基线。
- **沉淀到 wiki：** [`wiki/entities/paper-qwm.md`](../../wiki/entities/paper-qwm.md)

## 机构与作者

| 作者 | 机构 |
|------|------|
| Perry Dong, Chelsea Finn, Dorsa Sadigh | Stanford University |
| Yueru Jia | Peking University（亦与 Stanford 合作） |

## 开源状态（步骤 2.5，2026-09-21）

- **代码：** 项目页按钮 **「Code (coming soon)」** — 截至入库日 **待发布**。
- **论文 PDF：** [arXiv:2608.17163](https://arxiv.org/abs/2608.17163) 可公开获取。
- **关联链接：** 项目页引用 [EXPO-FT](https://pd-perry.github.io/expo-ft/) 作为 RL 微调背景。

## 项目页核心摘录

1. **核心问题：** 世界模型能否叠在标准 Q-learning 上 **只做 test-time scaling**，而学习仍 grounding 于真实在线数据？
2. **三步循环：** Imagine（世界模型预测候选动作后果）→ Evaluate（Q 函数打分）→ Learn（policy/critic 只用真实转移）。
3. **搜索树：** 策略采样候选 → 世界模型递归扩展 → Q 评估并聚合 → 选最高价值根动作；在线采样与评测均可启用。
4. **与 prior MBRL：** 不在 imagined trajectories 上训练，避免 compounding model bias。
5. **基座算法：** 主要展示 EXPO；亦展示 RLPD + QWM。
6. **基准：** Robomimic（state）与 LIBERO（pixel）；对比 model-free、TD-MPC2、EZ-V2。
7. **消融：** 在线+评测双阶段搜索、中等深度与 future-value 加权最优。

## 对 wiki 的映射

- [`wiki/entities/paper-qwm.md`](../../wiki/entities/paper-qwm.md)
