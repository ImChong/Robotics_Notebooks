# SPINE: Bridging the Cyber-Physical Gap with Agentic AI

- **作者：** Minkyu Ham, Dongho Kim, Chan Lee, Min Jun Kim, Yixi Zhang, Jiayi Wang, Han Liu
- **机构：** Northwestern University（Department of Statistics and Data Science; Department of Computer Science）
- **版本 / 载体：** arXiv:2607.13049v2 [cs.AI]（v1 提交于 2026-06-29，v2 修订于 2026-09-21；arXiv HTML 全文）
- **原文：** <https://arxiv.org/abs/2607.13049>（HTML 全文：<https://arxiv.org/html/2607.13049>）
- **项目页 / 代码：** 未确认开源代码（论文页首标示 `Website: magics-lab.github.io/SPINE-web`，但本地抓取上下文 `project_page_status: 0`、`github_urls: []`、`candidate_repos: []`，截至 2026-11-27 论文正文及缓存上下文中未见公开可访问的 GitHub 源码仓库链接）
- **核对范围：** 通读 `arXiv:2607.13049v2` HTML 全文全部章节（Abstract、I Introduction、II Related Work [II-A–II-C]、III SPINE System [III-A–III-B]、IV Experimental Design [IV-A–IV-C]、V Results [V-A–V-C]、VI Discussion、VII Conclusion、Acknowledgements 与 References [1]–[34]），以及全部图注与表格（Fig. 1–3、Table I 12 个注入故障场景、Table II 双平台定量对比、Table III DOBOT 消融实验）。
- **通读：** 已完成

## 原文核对笔记

1. **核心问题定义（Pre-operational Teleoperation Recovery）**
   - 论文将基础模型/VLA 与物理执行之间的驱动、网络接口、传感器、控制器及安全约束对齐层比作机器人的“脊髓（spinal cord）”。
   - 与针对已有可执行控制接口后失败恢复的工作（如 DROC、RACER、Self-Recovery Prompting、CoPAL、ASPIRE）不同，SPINE 面向 **pre-operational** 阶段，目标是把双臂平台恢复至 **teleoperation-ready**（主手可控制从手与夹爪且所需相机流可用），明确说明这不代表自主策略已就绪。

2. **系统架构（Claude Code Skills + Python Backend）**
   - **画像构建（`/spine-profile-build` + `/spine-init`）**：Python 编译器汇编源码/配置/命令关系证据包 → 9 个并行抽取子智能体（8 个按身份/组件/接口/软件/配置/流程/安全/诊断分域 + 第 9 个重构遥操作 DAG 命令图）→ 合并为 8 个分类 JSON + 1 个 manifest，并生成运行时契约、探针、硬件手册（hardware playbooks）与隐藏就绪检查（hidden readiness checks）→ 操作员确认并记录一次完整执行 → 9 个并行 curator 子智能体审查 + 1 个仲裁器消解冲突 + `profile doctor` 校验后封存。
   - **闭环调试（`/spine-debug` + `/spine-check` + `/spine-closeout`）**：`debug start` 加载画像并维护阻塞账本（blocker ledger）；4 个只读诊断子智能体并行分析终端输出、软件配置偏差、硬件可见性与就绪检查相关性，主智能体裁决“数字修复 vs 物理修复”；软件修改经 `debug after-fix` 触发再验证，物理操作每次仅指导 1 步硬件手册动作并经 `debug continue` 录入观测与运行探针；`debug finish` 要求最近一次干预后在同一最新轮次内同时通过实时遥操作管线（默认 10 s 观测窗）与隐藏就绪检查，且无剩余阻塞项；`/spine-closeout` 将有通过记录支撑的修复写入事件记忆。

3. **实验与消融边界**
   - 双平台共 12 场景：DOBOT X-Trainer（7 场景：D-SW1/2、D-HW1/2/3、D-HY1/2）与 AgileX PiPER（5 场景：P-SW1/2、P-HW1、P-HY1/2），基线为人工操作员使用同模型（Claude Sonnet 5.0）的标准 Claude Code。
   - DOBOT（SPINE $n=3$ vs 人工 $n=7$）：整体 OSS 从 $0.76 \pm 0.36$ 升至 $1.00 \pm 0.00$，平均 TTO 从 $14\text{m}51\text{s} \pm 10\text{m}44\text{s}$ 降至 $10\text{m}20\text{s} \pm 3\text{m}11\text{s}$（降幅 30%）；其中 D-HY1 上 SPINE 平均 TTO 为 $14\text{m}55\text{s}$，长于人工的 $11\text{m}55\text{s}$，但 OSS 为 $1.00$ 对 $0.64$。
   - PiPER（SPINE $n=3$ vs 人工 $n=9$）：整体 OSS 从 $0.99 \pm 0.07$ 升至 $1.00 \pm 0.00$，平均 TTO 从 $12\text{m}30\text{s} \pm 10\text{m}35\text{s}$ 降至 $7\text{m}48\text{s} \pm 3\text{m}27\text{s}$（降幅 38%），P-HW1 从 $13\text{m}36\text{s}$ 降至 $2\text{m}45\text{s}$（降幅约 80%）。
   - DOBOT 消融（Table III，消融组各场景仅 $n=1$ 试验）：移除 Profile 后整体 OSS 降至 $0.79$、平均 TTO 增至 $17\text{m}00\text{s}$；移除 Hardware Playbook 后整体 OSS 降至 $0.57$、平均 TTO 增至 $31\text{m}25\text{s}$。

4. **对 wiki 的映射**
   - 对 wiki 的映射：[SPINE: Bridging the Cyber-Physical Gap with Agentic AI](../../wiki/entities/paper-agentic-2607-13049.md)
