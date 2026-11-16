# Darwin Gödel Machine（DGM）官方实现

> 来源归档（ingest）｜代码入口和调用关系以官方仓库为准

- **类型：** 官方代码 / 开放式自改进编码智能体
- **仓库：** <https://github.com/jennyzzt/dgm>
- **论文 v3：** <https://arxiv.org/abs/2505.22954v3>
- **官方介绍：** <https://sakana.ai/dgm/>
- **开源：** Apache-2.0；Docker 与 Python 可运行入口已公开。
- **核查日期：** 2026-11-10
- **通读：** 已完成
- **对 wiki 的映射：** [DGM 论文实体](../../wiki/entities/paper-darwin-godel-machine.md)
- **此前来源：** 2026-10-28 已归档 README；本次扩展到 DGM_outer.py / self_improve_step.py 的关键函数及代码执行路径。

## 目录和执行入口

官方 README 提供 **python DGM_outer.py** 启动命令，执行前要求 Docker、Python 依赖、模型 API keys、SWE-bench checkout 或 Polyglot 数据准备。默认输出目录 **output_dgm/**。

| 源码 | 关键函数或路径 | 可验证作用 |
|---|---|---|
| [DGM_outer.py](https://github.com/jennyzzt/dgm/blob/main/DGM_outer.py) | main, initialize_run, choose_selfimproves, filter_compiled, update_archive | 初始化归档；从版本档案选亲本与任务；ThreadPoolExecutor 并发调用 self_improve；保留有效子代并写入 dgm_metadata.jsonl |
| [self_improve_step.py](https://github.com/jennyzzt/dgm/blob/main/self_improve_step.py) | self_improve, diagnose_problem, run_harness_swe, run_harness_polyglot | 读取任务、创建 Docker、还原亲本 patches、LLM 诊断、运行编码 Agent、自改补丁评测与 metadata.json 保存 |
| [coding_agent.py](https://github.com/jennyzzt/dgm/blob/main/coding_agent.py) | --self_improve（由上层通过命令行调用） | 初始被进化编码智能体；其程序代码由自改进得到的补丁迭代 |
| swe_bench / polyglot | harness | 调用独立 benchmark 执行实际编程任务，取得成功/失败结果 |
| utils/docker_utils.py | build_dgm_container、cleanup_container | 创建/清理容器和跨容器传递文件 |
| utils/evo_utils.py | get_model_patch_paths、get_all_performance、is_compiled_self_improve | 祖先补丁、评测汇总和入档有效性辅助 |

## 调用链（核查源码）

1. **DGM_outer.py main** 初始化初始档案与分数、按准确率和子代探索程度筛选亲本。
2. **ThreadPoolExecutor** 并行调用 **self_improve(parent_commit, entry, ...)**。
3. **self_improve** 创建容器、依次应用祖先 patch，调用 **diagnose_problem** 由 LLM 归纳当前失败任务。
4. 容器执行 **python /dgm/coding_agent.py --problem_statement ... --self_improve**；生成 **model_patch.diff**、自改记录；若补丁为空则不继续评测。
5. 对非空 patch 调用 **run_harness_swe** 或 **run_harness_polyglot**，记录报告与 overall_performance。
6. 外层 **filter_compiled** 过滤不具备有效代码修改能力的子代，然后 **update_archive**；默认策略 **keep_all** 保存有效子代，允许得分比父代低。

## 阅读与运行边界

- **已读：** README、DGM_outer.py、self_improve_step.py，及它们引用的可公开源码接口/任务目录；此归档记录运行链，不声称逐行审计所有第三方 benchmark 或每一个生成补丁。
- **未执行：** 高成本模型请求、Docker 中生成不受信任代码、完整 SWE-bench/Polyglot 实验、结果重现。
- **安全：** 官方 README 明确警告**执行由模型生成的不受信任代码**有破坏性风险。实验所用隔离与监督不应被视为生产级安全证明。
- **版本：** 官方 GitHub main 代码路径核对日 2026-11-10；论文原文是 arXiv v3。
