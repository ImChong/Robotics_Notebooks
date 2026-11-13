---
type: entity
project_id: embodied-first-exam
arxiv: "2611.12177"
original:
  - ../../sources/papers/embodied-first-exam-2611-12177.md
tags:
  - benchmark
  - coding-agents
  - embodied-ai
  - evaluation
  - harbor
  - robotics
  - simulation
  - open-source
status: complete
updated: 2026-11-11
related:
  - ./rle-bench.md
  - ../overview/hub-embodied-eval-benchmark.md
  - ../concepts/simulation-evaluation-infrastructure.md
sources:
  - ../../sources/papers/embodied-first-exam-2611-12177.md
  - ../../sources/sites/embodied-first-exam-ai.md
  - ../../sources/repos/embodied-first-exam.md
summary: "Embodied First Exam 论文评测 35 套件、1156 任务、47 种机器人，论文标准核心 456 任务的最强模型成功率34%；当前仓库已扩充到36套件和1211任务，需区分论文口径。"
---

# Embodied First Exam（具身代码代理考核）

**Embodied First Exam**（[项目页](https://www.embodied-first-exam.ai/)，[GitHub](https://github.com/mll-lab-nu/embodied-first-exam)，**MIT**）把机器人任务环境作为 coding agent 的考场：agent 在固定场景里编写程序、运行并查看结果，提交结果后由独立沙箱验证。项目通过 [Harbor](https://github.com/laude-institute/harbor) 运行；它衡量的是 agent 在具身任务中的程序化交互能力，不是单独比较某个机器人策略模型的成功率。

## 一句话定义

**Embodied First Exam 把「coding agent 能否写出在机器人任务环境中正确运行的程序」变成一套可用 Harbor 批量执行、由独立沙箱验证的考核。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Harbor | Harbor evaluation framework | Laude Institute 沙箱评测运行时，用于批量运行 agent 任务 |
| MIT | MIT License | 代码仓许可；第三方模拟器、资产和基础镜像另有许可 |
| GPU | Graphics Processing Unit | 部分仿真套件需要相应的 NVIDIA GPU 环境 |

## 核心信息

| 项 | 内容 |
|----|------|
| 对象 | Coding agents |
| 运行框架 | Harbor |
| 论文规模 | 2026-11 arXiv：35 suites、1,156 unique tasks、47 种机器人；标准核心 456 任务 | 
| 论文核心结果 | 最强模型在 456 个核心 Standard 任务成功率 34%（论文作者报告） |
| 当前代码 README | 36 benchmark / 1,211 unique tasks / 2,398 task directories；核心配置 498 tasks，较论文口径增加 HumanCLAW 等；**不应改写论文分母** |
| 任务闭环 | 编写程序 → 运行任务 → 查看结果 → 独立沙箱验证 |
| 运行环境 | Linux x86_64；Docker / Compose v2 / BuildKit；GPU 依 benchmark 而定 |
| 代码许可 | MIT；第三方模拟器、资产和基础镜像仍遵循各自许可 |
| 发布日期 | 2026-10-19（官方 GitHub README） |

## 为什么重要

- **评测具身 coding agent：** 任务要求代理通过程序与机器人任务环境交互，再根据执行反馈继续处理；它关注任务执行过程，不只是静态问答或代码补全。
- **汇集多种任务套件：** 将不同 benchmark 的机器人任务放入 Harbor 可运行、可验证的评测入口，便于使用同一 agent runner 批量执行。
- **把运行与验证隔开：** 任务结果由独立 sandbox 检查，有助于评测端复核产物和任务完成情况。
- **容易与 RLE-Bench 混淆：** [RLE-Bench](./rle-bench.md) 明确按机器人学习工程师的多类工作流组织资格考；Embodied First Exam 的重点是汇集多套冻结任务，考核 coding agent 在这些任务中的程序执行表现。两者对象相近，但任务设计和覆盖范围并不相同。

## 流程总览（论文评测协议）

```mermaid
flowchart TD
  T["35 套件 / 1156 论文任务"] --> C["冻结任务场景与各自机器人"]
  C --> A["Coding agent 根据准许观测编写控制程序"]
  A --> S["Harbor 在任务环境中执行"]
  S --> O["反馈：Standard / Realtime / Privileged 按协议区分"]
  O --> A
  S --> V["独立 verifier / grader"]
  V --> R["论文核心 Standard 成功率统计"]
```

**Standard** 为论文主要评测设置；**Privileged** 具有更丰富状态/重置能力；**Realtime** 关注时限。不能跨模式合并统计。

## 实验与评测

论文报告 **35 个 suites、1,156 项任务、47 种机器人**；筛选的 **456 个核心 Standard 任务**上最强模型成功率 **34%**。这是作者报告的 agent 评测，非本库独立复现。

官方当前 README 已增至 **36 个 benchmark、1,211 项任务（2,398 个 task dirs）**，核心为 **498 个任务**；README 注明论文/榜单的 1,156 口径不含 HumanCLAW，因此保留两个版本数字。

## 结论

**该工作将“代码代理能不能让机器人完成任务”从文本评测转为可执行且分权限的标准化环境，但目前高难度具身任务成功率仍有限。**

1. 始终在 Standard / Realtime / Privileged 相同协议内比较。
2. 区分论文 1,156/456 和扩充版仓库 1,211/498 的分母。
3. 仿真成功率不能直接替代真机性能、能耗和安全评测。
4. 验证需要 Harbor、对应 benchmark 资产与各自许可。

## 与其他工作对比

本基准评价 agent **执行机器人任务**，不是训练一个统一操作策略。可用它比较[Claude Fable 5.2](./claude-fable-5-2.md)的独立仿真指标叙事，也可作为研究[SWAM](./paper-swam-cross-modal-alignment.md)、[WICL](./paper-world-in-context-learning-roboscript.md)、[Compass 1](./paper-compass-1-navigation.md)与[Flip Anything](./paper-flip-anything-agentic-self-learning.md)时理解任务与成功判据差异的参照；以上不同工作当前**没有相同评测协议**，不可直接排名。

## 工程实践

参照官方 README 安装 Harbor，分别选择 core/all 数据集，留存 task ID、模式、模型、agent、试验次数、GPU 与 verifier 版本；公开仓库已扩充，需要锁定 commit 再对照论文报告。

## 快速开始（官方 README）

~~~bash
git clone --recurse-submodules https://github.com/mll-lab-nu/embodied-first-exam.git
cd embodied-first-exam
uv tool install --force -e . --with-executables-from harbor

# 核心任务
harbor run -d embodied-first-exam -a codex -m <model>

# 全量任务
harbor run -d embodied-first-exam@all -a codex -m <model>
~~~

运行前需按具体 benchmark 准备数据、GPU 驱动和磁盘空间。部分任务首次运行会构建镜像；项目 README 说明构建过程可能接受其安装组件的许可条款。

## 源码运行时序图

入口：`efe run -a codex -m <model>`（`efe` = `scripts/efe.py:main`，其余参数交给 Harbor；等价于 `harbor run -d embodied-first-exam -a codex -m <model>`）。下图以 standard 模式的 behavior-1k 任务为例（sim 边车 + agent 容器两段式）。

```mermaid
sequenceDiagram
    participant E as efe.py run
    participant H as Harbor 0.23.0 harbor.cli
    participant SIM as sim 边车 behavior-1k/sim/server.py
    participant L as rcb_limited.serve HTTP 8000
    participant M as main 容器 agent codex
    participant C as rcb_client.Robot
    participant V as verifier tests/test.sh

    E->>E: run(argv) 选任务集与模式，生成 job 配置
    E->>H: harbor(["run", "-c", cfg, ...])
    H->>SIM: compose 启动 sim，command sh /srv/run.sh
    SIM->>SIM: run.sh 执行 python -u /srv/server.py
    SIM->>L: L.serve(Service(submit=do_submit, observe=do_observe, ...))
    H->>M: 启动 agent 容器，agent 在 /app 中写代码
    M->>C: Robot() 连接 SIM_URL / sim:8000 / localhost:8000
    C->>L: GET /spec 直到服务就绪
    L-->>C: spec（protocol eai-standard 2.x）
    M->>C: robot.submit(type, actions)
    C->>L: POST /submit（application/x-rcb1）
    L->>L: parse_request 校验后调用 svc.submit
    L->>SIM: do_submit 校验字段（非法则 400，不执行）
    SIM->>SIM: run_joint_position / run_ee_pose 逐行调用 ctrl_step
    SIM->>SIM: check_success 命中则 finish success
    SIM-->>L: out 与录像 parts
    L-->>C: 响应 encode_response
    C-->>M: Result（meta 与帧）
    M->>C: get_obs(seconds) 观察，循环直至 cutoff
    C->>L: POST /observe
    L->>SIM: do_observe 保持 hold 动作推进仿真
    H->>SIM: 收集钩子 finalize.sh（结束 live episode，写 /out）
    H->>V: 上传 sim 的 /out 记录，在独立沙箱运行
    V->>V: rcb_limits.py sanitize /app/output
    V->>V: python /tests/grade.py 评定 live episode
    V-->>H: /logs/verifier 结果
```

图依据：mll-lab-nu/embodied-first-exam@c7b94c7，实际读过 `README.md`、`pyproject.toml`、`docs/protocol/README.md`、`scripts/efe.py`（run / main）、`embodied_first_exam/standard/run.sh`、`embodied_first_exam/standard/rcb_limited.py`（serve / bind / make_handler 的 do_POST）、`embodied_first_exam/standard/v2_4/rcb_client.py`（Robot / _post / submit / get_obs）、`embodied_first_exam/benchmarks/behavior-1k/sim/server.py`（do_submit / do_observe / ctrl_step / 入口）、`tasks/behavior-1k/behavior-assembling-gift-baskets-i00-standard/environment/docker-compose.yaml`、`tests/test.sh`、`embodied_first_exam/benchmarks/behavior-1k/sim/finalize.sh`（前若干行）。Harbor 本体（0.23.0）是外部依赖，未读其源码，图中 Harbor 内部调度只按仓库注释与调用关系画。

## 局限与风险

- 论文主要测量仿真环境里的 coding agent 编程与控制能力；与具身动作策略模型、真实机器人闭环和安全部署的评价对象不同。
- 部分任务没有完整参考解，成功标准依赖各 suite verifier；单次尝试与 agent 运行预算影响统计。
- 新增套件后仓库总数与论文口径不同，论文正文与附录 A–K 已核对；完整仓库复现与多次重复评测尚未执行。

## 关联页面

- [RLE-Bench](./rle-bench.md) — 同样通过 Harbor 评 coding agent，但任务设计与覆盖范围不同
- [具身评测基准选型闭环](../overview/hub-embodied-eval-benchmark.md) — 本页所在的评测选型知识链

- [Claude Fable 5.2](./claude-fable-5-2.md) — 另一类仿真机器人能力评测
- [SWAM](./paper-swam-cross-modal-alignment.md) — 视频和动作联合学习
- [WICL / RoboScript](./paper-world-in-context-learning-roboscript.md) — 具身 agent 操作
- [Compass 1](./paper-compass-1-navigation.md) — 长时导航
- [Flip Anything](./paper-flip-anything-agentic-self-learning.md) — 自迭代机器人技能

## 参考来源

- [arXiv:2611.12177 论文核验摘录](../../sources/papers/embodied-first-exam-2611-12177.md)
- [官方项目页](../../sources/sites/embodied-first-exam-ai.md)
- [官方代码仓与运行说明](../../sources/repos/embodied-first-exam.md)
- [RLE-Bench](./rle-bench.md) — coding agent 机器人学习工程能力评测
- [具身评测基准选型闭环](../overview/hub-embodied-eval-benchmark.md)
- 源码核对（2026-11-02）：mll-lab-nu/embodied-first-exam@c7b94c7，读过 README.md、pyproject.toml、docs/protocol/README.md、scripts/efe.py、embodied_first_exam/standard/{run.sh,rcb_limited.py,v2_4/rcb_client.py}、embodied_first_exam/benchmarks/behavior-1k/sim/server.py、tasks/behavior-1k/behavior-assembling-gift-baskets-i00-standard/{environment/docker-compose.yaml,tests/test.sh}、embodied_first_exam/benchmarks/behavior-1k/sim/finalize.sh（前若干行）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [Embodied First Exam — arXiv:2611.12177](https://arxiv.org/abs/2611.12177)
- [官方任务清单与 Quick Start](https://github.com/mll-lab-nu/embodied-first-exam)
