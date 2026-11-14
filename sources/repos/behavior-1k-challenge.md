# BEHAVIOR-1K / 2026 BEHAVIOR Challenge｜官方评测源码

> 来源归档；**与参赛者/冠军方案源码区分**。

- **类型：** repo / 2026 BEHAVIOR Challenge 评测与仿真环境
- **官网：** https://behavior.stanford.edu/challenge/index.html
- **GitHub：** https://github.com/StanfordVL/BEHAVIOR-1K
- **文档指定版本：** v3.9.3（2026-11-14 GitHub 文件 API 读取该 ref 失败；源码阅读基于 main）
- **数据说明：** https://behavior.stanford.edu/challenge/dataset.html
- **演示数据：** https://huggingface.co/datasets/behavior-1k/2026-challenge-demos
- **评测规则：** https://behavior.stanford.edu/challenge/evaluation.html
- **基线：** https://behavior.stanford.edu/challenge/baselines.html
- **原始 BEHAVIOR-1K 论文：** https://arxiv.org/abs/2403.09227（非 2026 Challenge 独立论文）
- **首次入库：** 2026-10-03；**复核日期：** 2026-11-14
- **核对范围：** 官方 Challenge 首页、Dataset、Baselines、Evaluation and Rules；StanfordVL/BEHAVIOR-1K main 评测 CLI、BatchedEvaluator、WebsocketClientPolicy/Server 与 score_utils 入口及连接逻辑。
- **通读：** 已完成

## 代码入口及运行流程

- 评测 CLI：OmniGibson/omnigibson/eval/eval.py，main() 解析 task、mode、实例、policy server、wrapper 等参数，组装 OmegaConf 与 BatchedEvaluator。
- 任务仿真 / rollout：OmniGibson/omnigibson/eval/evaluator.py，resolve_instance_ids 区分 train/public_test/hidden_test，BatchedEvaluator.run() 执行并收集结果。
- 策略通信：OmniGibson/omnigibson/eval/policies.py 与 eval/utils/network_utils.py，WebsocketPolicy 使用 healthz 和 websocket 调用策略服务；obs → msgpack action。
- 指标读取与统计：OmniGibson/omnigibson/eval/utils/score_utils.py、eval/metrics 与 TaskMetric；输出 JSON 中有 q_score、time、agent_distance 等字段。
- 启动顺序：先运行兼容的 policy server，再调用 python -m omnigibson.eval.eval --task-name turning_on_radio --host 127.0.0.1 --port 8000 --instance-indices 0 --num-rollouts 1 --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper --write-video。
- 完整依赖、资产和数据准备详见官方 v3.9.3 文档及 setup.sh；本次**未实际完成安装或运行**。

## 开源边界与版本差异

赛事环境与评测工具公开，训练示范通过 Hugging Face 独立提供，大型数据/场景资源需额外下载。2026-11-14 的 GitHub 文件接口对于 ref=v3.9.3 返回未找到，不能据此把 main 核对结果等同于官方指定 tag 的行为；先核实可访问的对应 tag/commit，才能严谨完成逐行复现验证。

此仓库不是 HiPhys-RSI 冠军的完整开源代码；冠军项目页未给出其系统完整运行仓库。

## Wiki 映射

- [2026 BEHAVIOR Challenge](../../wiki/entities/behavior-challenge-2026.md)
- [BEHAVIOR-1K 基准](../../wiki/entities/behavior-1k.md)
- [HiPhys-RSI 冠军方案](../../wiki/entities/paper-hiphys-rsi.md)
- [赛事官网归档](../sites/behavior-challenge.md)
