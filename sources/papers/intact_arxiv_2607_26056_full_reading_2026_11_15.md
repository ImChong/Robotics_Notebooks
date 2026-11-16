# INTACT：原文通读与实现复核（2026-11-15）

- **类型：** paper
- **原文：** <https://arxiv.org/html/2607.26056>
- **版本：** 2026-11-15 获取的 HTML；具体正文、附录和代码范围如下。
- **代码：** <https://github.com/zju3dv/INTACT-JEPA>
- **核对日期：** 2026-11-15
- **核对范围：** arXiv HTML 正文 §1–7、附录 A–L、全部表格与图注；另通读规范仓当前 README、项目页及 scripts/train.sh、scripts/eval.sh，核对 evaluation.py 的 run 主流程与因果历史接入段。论文数值与 2026-09-14 后的实现修订分别记录；未运行训练或机器人实验。
- **通读：** 已完成

本次复核单独存档；[历史摘录](intact_arxiv_2607_26056.md)保留原有日期和发布状态，不能代替下述版本及协议核对。

## 原文与实现核对

### 方法与支持范围

- 局部意图 `z[t+1]-z[t]` 保留后继梯度；目标意图 `sg(z[g])-z[t]` 对目标停梯度。共享四槽输入与动作分布监督提供动作律等价，没有两个端点间的直接 L2 匹配，也没有证明整个表征空间全局线性。目标分支与目标条件行为克隆有联系，仍保留前向 JEPA 与 SIGReg。
- Direct 不采样候选、不计算终点规划代价，但通过前向模型递归生成 H=5 个动作块，执行首个动作块后重规划；不能把“无搜索”写成“无前向预测”。Guarded 的 128×3=384 是采样候选预算，另保留 Direct/历次最优，最终均值重评分也可能额外调用模型；与 300×30=9000 的 CEM 比较时须注明计数对象。
- 单任务一 epoch，3 个训练 seed，各在 0/1/42 三个评测 seed 上运行，每次 100 episodes。共享编码器实验训练 5 epochs，仍使用已知任务身份与任务专属动作头，并非跨任务单一无条件策略。

### 论文快照与修订实现不能混报

| 口径 | 官方单任务 Direct | 官方 Guarded A | 共享 E5 Direct | Direct 规划时延 |
|------|------------------|----------------|----------------|-----------------|
| 本次通读的 arXiv HTML | 95.33±0.58% | 96.86±0.38% | 89.39±0.77% | 2.9–5.5 ms |
| 当前 README / 项目页，标明因果历史修复后 | 95.61±0.59% | 96.58±0.44% | 91.22±0.51% | 3.9–4.8 ms |

修复说明指出 actor-backed 评测曾重复输入前一步动作；当前 `run` 用 `build_initial_history` 初始化 `rows[t-slots:t]`，只在 episode 起点之前左填 raw zero，之后由 `StatefulActionHistoryPolicy` 移入实际执行的 primitive actions。当前行 action 和目标 action 都不可见。history-free 的 94.25±0.08% 是单独消融，不是 Guarded 成绩。上述结果均为作者报告，未在本地重测。

- 论文 CLEAR-LeWM v0.5.1 Moderate 下，Direct / Guarded 为 83.33 / 84.83%，低于官方 95.33 / 96.86%；Reacher 角度误差判据改变后 Direct 为 49.56%，不能合并平均。当前仓库另外提供 CLEAR v0.8 适配入口，须固定 evaluator 和 checkpoint revision。
- 动作头关闭的共享 E5 CEM 由 66.17% 升至 70.08%，支持表征也有改善，但 kNN/CKA 的相关性不证明任意反事实目标可达；三训练 seed、示范支撑、多峰动作被高斯均值平滑仍是边界。论文时延仅为规划器计时，不是传感、网络与机器人控制的端到端周期。

### 当前代码入口

- `scripts/train.sh` 调用 `train.py` 与 Hydra 配置，smoke 为单批次、batch size 2 的连通检查；数据须自行准备，不隐式下载，也不覆盖既有实验目录。
- `scripts/eval.sh` 映射 `direct / pure_cem / guarded_a`，强制官方协议与显式 inference mode，默认 math SDPA；`evaluation.py:run` 加载模型、标准化器、solver 与环境，在评测前安装状态化因果历史包装。
- 论文冻结 checkpoint `paper-e5-goal-v1` 使用配套 `paper_runtime/` 兼容环境；不能拿旧 checkpoint 与当前实现混合后称为严格复现。

## 对 wiki 的映射

- [INTACT 详情](../../wiki/entities/paper-intact.md)
