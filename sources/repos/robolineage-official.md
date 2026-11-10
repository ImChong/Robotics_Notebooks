# RoboLineage — 官方开源仓库及运行时源码核验

- **仓库：** <https://github.com/robolineage/robolineage>
- **官方归属：** RoboLineage 项目主页 Code 链接所指组织与仓库。
- **协议：** MIT（仓库 LICENSE / README）。
- **核查日期：** 2026-11-10
- **论文：** <https://arxiv.org/html/2606.22142v1>
- **项目：** <https://robolineage.github.io/>
- **核对范围：** README、AGENTS.md、run.sh、src/robolineage_app/__main__.py、src/robolineage_app/runtime.py 的 UnifiedRuntime/委托结构、src/robolineage_session/dispatcher.py、src/robolineage_post_rollout/worker.py 与 formal_review.py、src/robolineage_dataset/lock_writer.py、src/robolineage_train/lifecycle.py、src/robolineage_eval/agents.py；以及 docs/、configs/、examples/、scripts/ 的仓库目录与 README 说明。
- **通读：** 已完成（上述核心入口与本节点关注的执行路径；不代表逐行审计整个约 185k 字符的 runtime.py）。
- **公开范围：** Python 研究运行时 + ROS2 robot profile 和采集适配 + 在线 VSA + 异步 review / data governance + schema + dataset lock + 训练适配 + policy eval + FastAPI / Vue 前端 + tests。
- **尚未完成：** README checklist 的无实体硬件 mock replay、打包 benchmark scoring、更多 adapter/机器人 profile/本地模型路由。**未**独立在 ROS2 workstation 或真机运行。

## 源码入口与模块证据（对应用于 wiki 的 sequenceDiagram）

| 真实文件/符号 | 已核对行为 | 注意边界 |
|---|---|---|
| `run.sh` | 装载 `ROBOLINEAGE_CONFIG` / ROS 环境，启动 `python -m robolineage_app`，并可启动 Vue Vite frontend | README 说明 Ubuntu ROS2；`rclpy` 用 ROS 发行版环境提供 |
| `src/robolineage_app/__main__.py:main` | 参数解析、`load_config`、`UnifiedRuntime` 配置、FastAPI session/health uvicorn servers | 端口缺省 8080/8081 |
| `src/robolineage_app/runtime.py:UnifiedRuntime` | 编排 robot、raw data source、online VSA、session、post-review、training/evaluation 代理模块 | 这里是编排层，不是策略本身训练网络 |
| `src/robolineage_session/dispatcher.py:FeedbackDispatcher.on_snapshot` | VSA 的 phase/progress/risk 以事件广播与记录，高风险触发风险事件 | 异步语义反馈不替代硬件安全系统 |
| `src/robolineage_post_rollout/worker.py:PostRolloutReviewWorker._loop` | FIFO worker 取 rollout path，实例化 `PostRolloutReviewAgent` 并调用 `.run(rollout_dir)` | 离线审阅独立于 raw collection |
| `src/robolineage_post_rollout/formal_review.py` | EvidenceBundle、annotation、failure/admission 的 typed schemas 与 validated artifact writer | VLM 的 admission 字段不直接被信任 |
| `src/robolineage_train/lifecycle.py:TrainingLifecycleRunner.run` | `adapt_data` → `train_adapted`：`build_train_manifest_from_post_review`、Data Health → `DatasetUpdater.update` → `FrameworkAdapter.adapt_dataset` → `FrameworkAdapter.run_training_only` | 某一 backend 的训练逻辑在外部现有 learner，非内置万能模型 |
| `src/robolineage_dataset/lock_writer.py:DatasetLockWriter.write` | 已存在锁拒绝覆盖，`dataset.lock.tmp` 原子替换为只读 `dataset.lock` | 代表文件级 immutable lock，不是分布式强一致性承诺 |
| `src/robolineage_eval/agents.py:PolicyEvaluationAgent.run` | `EvidenceBuilder.build` + `AnnotationAgent.annotate` + `FailureAnalysisAgent.analyze` → 写入 `policy_evaluation.json` 和 collection recommendation | 评估输出需要现有任务验收与操作员确认 |

## README Quickstart（官方原样语义）

```bash
git clone https://github.com/robolineage/robolineage.git
cd robolineage
python3 -m venv .venv-ros
source .venv-ros/bin/activate
python -m pip install -e ".[dev,contracts,data-source,vsa,session,agents,dataset,train]"
cd frontend && npm install && cd ..
export ROBOLINEAGE_CONFIG=configs/robolineage_default.yaml
./run.sh
```

按官方文档，默认 backend localhost:8080、health:8081、frontend:5173。通过 `python -m robolineage_robot.topic_probe --ros-domain-id ${ROS_DOMAIN_ID:-0}` 可辅助 ROS2 profile 绑定；用 `ROBOLINEAGE_CONFIG=configs/arx_one.yaml ./run.sh` 启动指定 profile。以上是**作者提供的可运行入口**，本次是静态代码阅读，没有运行真机。正式运行必须审查机器人 safety case、topic binding 与 API secrets。

## 对论文指标的复现约束

- 本仓说明 release 中存在 schema、prompt、sample artifact 和 runtime tests；它**未**声称已公开用于复现论文 25-task/500-rollout 全部标注录像、全部模型 API 配置和实机环境。
- README 列 packaged scoring scripts 和 mock replay pipeline 为待办，故不能写「开箱可复现 96.4% 或 58.3%」。
- 代码许可为 MIT；外部 LLM/VLM API、机器人驱动、已有训练器与数据仍各受自身许可和访问限制。

## 对 wiki 的映射

- [唯一主实体](../../wiki/entities/paper-robolineage.md)
- [原论文读后核验](../papers/robolineage_arxiv_2606_22142.md)
- [官网记录](../sites/robolineage-project.md)
