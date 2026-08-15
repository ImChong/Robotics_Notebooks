# Temporal GRPO（arXiv:2608.13026）

> 来源归档（ingest）

- **标题：** Temporal GRPO: Beyond Trajectory-Level Credit in Vision-Language-Action Reinforcement Learning
- **短名：** Temporal GRPO
- **类型：** paper / vla / reinforcement-learning / grpo / credit-assignment
- **arXiv：** <https://arxiv.org/abs/2608.13026>
- **PDF：** <https://arxiv.org/pdf/2608.13026>
- **HTML：** <https://arxiv.org/html/2608.13026>
- **项目页：** 无
- **代码：** 论文未列仓库；检索未见官方实现 → **确认未开源**
- **作者：** Yao Zhou、Hang Gao、Fengge Wu、Changwen Zheng、Wenwen Qiang（通讯）
- **机构：** 中国科学院软件研究所（ISCAS）。预印本作者行未写单位；按通讯作者主页与长期合作关系归档（Wenwen Qiang / Changwen Zheng / Fengge Wu 均属 ISCAS）
- **版本：** arXiv:2608.13026（2026-08-13）
- **入库日期：** 2026-08-15
- **一句话说明：** 结果驱动 VLA-RL 里，整条轨迹共用一个成败优势会惩罚已经做对的前序阶段（trajectory-level credit aliasing）。Temporal GRPO 用可检测阶段对齐动作区间，只在「进入同一阶段」的 rollout 之间比相对优势，并写回对应区间。

## 摘要级要点

- **问题：** 常见 GRPO 后训练把 \(\widehat{A}_i\) 广播到轨迹每一步。早失败和「前几段都对、最后放置失败」拿到同一个失败优势，前序正确动作被一起压掉。
- **方法：** 冻结 RynnBrain-4B 提语义阶段 → Stage Compiler 编成有序、可检测的信用阶段 → 仿真特权状态对齐区间 → 只比较进入该阶段的 rollout → 阶段优势只写回 \(B_{i,k}\)。仍是一条 VLA、一次 GRPO、完整 rollout。
- **不是：** 分阶段独立训子策略；也不是只把阶段进度加成标量再整轨广播（那是文中 Stage-Reward GRPO）。
- **RoboTwin 2.0：** 同一 OpenVLA-OFT SFT 热启动、同一预算。宏平均 **75.8±0.7**，比最强对照 SimpleVLA-RL **68.8** 高 7.0 点；短/中/长+超长分别 +8.3 / +6.5 / +6.2。
- **LIBERO-Long：** 按首次分歧阶段 \(m_d\) 对齐：Trajectory-GRPO 伤前序完成率，Temporal GRPO 前序 \(\Delta p_k\approx 0\)、增益集中在 \(m_d\)。全文 99.1±0.4 vs Trajectory-GRPO 88.4。
- **开源（截至 2026-08-15）：** 无项目页、无 GitHub；阶段谓词与特权状态仅训练期使用。勿与 TGRPO（arXiv:2506.08440，[hahans/TGRPO](https://github.com/hahans/TGRPO)）或图像生成 TempFlow-GRPO 混名。

## 核心摘录（面向 wiki 编译）

### 信用改写

- 阶段参与：\(V_{i,k}=1\) 当且仅当 \(k=1\) 或已完成 \(m_{k-1}\)；没进入的 **不当成该阶段失败**。
- 阶段结果 \(R_{i,k}\in\{0,1\}\)；末段 \(m_K\) 等于任务成功。
- \(\widehat{A}_{i,k}\) 只在 \(V_{i,k}=1\) 的组内标准化；全成功或全失败则本步跳过该阶段。
- \(\widehat{A}_{i,t}=\sum_k \mathbb{I}[t\in B_{i,k}]\widehat{A}_{i,k}\)；区间不重叠，每步至多一个优势。动作块继承所属区间。

### RoboTwin 2.0（Table 1，三种子）

| 方法 | Short | Medium | Long & Extra-Long | Macro |
|------|-------|--------|-------------------|-------|
| OpenVLA-OFT (SFT) | 21.3 | 47.1 | 46.5 | 38.3 |
| Trajectory-GRPO | 37.8 | 52.6 | 48.7 | 46.4 |
| TGRPO | 43.9 | 58.4 | 54.1 | 52.1 |
| Stage-Reward GRPO | 52.7 | 64.2 | 60.8 | 59.2 |
| SimpleVLA-RL | 64.9 | 72.5 | 69.0 | 68.8 |
| **Temporal GRPO** | **73.2** | **79.0** | **75.2** | **75.8** |

### LIBERO-Long 消融（Table 2）

| 变体 | SR (%) |
|------|--------|
| Temporal GRPO | **99.1±0.4** |
| w/o Stage Compiler | 96.8 |
| Stage-Reward GRPO | 94.7 |
| w/o entered-stage gating | 92.5 |
| w/o same-stage grouping | 90.6 |
| Trajectory-GRPO | 88.4 |

最大组件跌幅来自「取消同阶段编组」：不同阶段的动作区间不能放进同一个相对比较组。

### 开源核查（步骤 2.5）

无项目页。论文未承诺放代码，也未列 GitHub / HF。阶段检测用仿真特权状态，评测时不用。→ **确认未开源**。未建 `sources/repos/` / `sources/sites/`。

## 对 wiki 的映射

- 升格 [Temporal GRPO 论文实体](../../wiki/entities/paper-temporal-grpo.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[TEMPO](../../wiki/entities/paper-tempo.md)、[Green-VLA](../../wiki/entities/paper-greenvla-staged-vla-humanoid.md)、[RoboTwin](../../wiki/entities/robotwin.md)、[LIBERO](../../wiki/entities/libero-benchmark.md)、[OpenVLA](../../wiki/entities/openvla.md)、[WCM](../../wiki/entities/paper-wcm-world-critic-model.md)、[RynnBrain 1.1](../../wiki/entities/paper-rynnbrain-1-1.md)

## 当前提炼状态

- [x] 别名问题、阶段机制、两套数字、未开源结论
- [x] wiki 实体与交叉引用
