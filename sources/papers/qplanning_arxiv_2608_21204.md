# Q-Planning（冻结 BC 的离策略 Q 函数自改进）

> 来源归档（ingest）

- **标题：** Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.21204>
  - <https://varungiridhar.github.io/qplanning/>
- **代码：** <https://github.com/varungiridhar/qplanning-code>
- **机构：** 佐治亚理工学院（Georgia Tech）
- **入库日期：** 2026-08-25
- **一句话说明：** 为大型 visuomotor BC/VLA 策略配小型离策略 Q 函数：推理时对冻结 BC 的 N 个 action chunk 做 Q 加权平均选动作；在线只微调 Q、吸收成功与失败 rollout，实现无需人类演示的自改进。

## 核心摘录（MVP）

### 1) BC 与 Q 的数据不对称

- **摘录要点：** BC 只能模仿成功演示；Q 估计价值而非动作，可同训演示、再吸收部署期成功与失败轨迹。据此保持多十亿参数 BC **冻结**，仅更新约 1B 参数的 Q。
- **对 wiki 的映射：**
  - [Q-Planning](../../wiki/entities/paper-qplanning.md) — 核心动机与自改进环。
  - [VLA](../../wiki/methods/vla.md) — 大模型 BC 先验 vs RL 微调代价对照。

### 2) Q-chunking + HL-Gauss + 实时 Q 加权规划

- **摘录要点：** 将长度 H 的 action chunk 视为 super-action；Q 网络自有 DinoV2+T5 编码器，HL-Gauss 分类回归稳定长视界价值；推理从冻结 BC 截断 3 步 flow 采样 N 个候选，softmax(Q/λ) 加权平均执行（非 argmax）。RoboTwin N=32 规划 **400 ms/步**（<960 ms 重规划预算）。
- **对 wiki 的映射：**
  - [Q-Planning](../../wiki/entities/paper-qplanning.md) — 架构与延迟剖面。
  - [Action Chunking](../../wiki/methods/action-chunking.md) — chunk 级 Q 与重规划节拍。

### 3) 仿真与真机自改进结果

- **摘录要点：** 10 轮在线迭代：LIBERO-10 **93→99%**、RoboTwin **83.8→91.4%**；近天花板套件缩短成功 episode 长度。双臂真机 stack-cups **40→90%**、insert-wallet **25→80%**（5 轮，BC 冻结）；filtered SFT 仅成功轨迹停滞于 55%/30%。同预算下唯一稳定从失败学习、且无辅助 actor 的方法。
- **对 wiki 的映射：**
  - [Q-Planning](../../wiki/entities/paper-qplanning.md) — 评测表与对照定位。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 双臂接触丰富任务语境。

### 4) 开源状态（截至 2026-08-25）

- **摘录要点：** **已开源**：`qplanning-code` 提供 `qplanning` CLI（eval / self-improve / train-q / report）、LIBERO 与 RoboTwin 配置；默认 BC 为 **FastWAM**，策略视为满足「可采样 chunk」接口的黑盒。
- **对 wiki 的映射：**
  - [qplanning-code](../../sources/repos/qplanning_code.md) — 仓库布局与复现入口。
  - [Q-Planning 项目页](../../sources/sites/qplanning-varungiridhar.md) — 步骤 2.5 核查。

## 当前提炼状态

- [x] arXiv + 项目页 + GitHub README 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-qplanning.md` 新建
