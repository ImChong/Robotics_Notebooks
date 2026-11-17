# mind-palace-laeqa/benchmark 官方仓库归档

- **仓库：** <https://github.com/mind-palace-laeqa/benchmark>
- **许可证：** MIT（仓库 LICENSE）
- **类型：** 算法参考实现 / LA-EQA Benchmark
- **论文：** <https://arxiv.org/abs/2507.12846>
- **项目主页：** <https://mind-palace-laeqa.github.io/>
- **核验日期：** 2026-11-17
- **核对范围：** README 安装与评测说明、scripts/run_la_eqa_mind_palace_exploration.ipynb，以及 src/contextual_long_term_reasoning/ 的 mind_palace_generation.py、mind_palace_exploration.py、eqa_reasoning.py、belief_manager.py、eqa_evaluation.py、openai_interface.py 之间的导入与 notebook 中的主要调用；只作静态源码阅读，未执行外部 API 或机器人平台。

## 核验到的源码入口

1. README：```conda env create -f environment.yaml```、```conda activate laeqa```、```pip install -e .```；Python ≥3.9；可选安装 Recognize Anything Model。
2. ```scripts/run_la_eqa_mind_palace_exploration.ipynb```：选单题使用 ```test_one_question = True```，改 False 可遍历题集；从 ```la_eqa_benchmark/eqa-questions.json``` 加载问答，并读入 episode 信息与场景资源。
3. ```mind_palace_generation.py```：```SceneGraph```、```LoadingHabitatSceneGraph```、```LoadingIsaacSceneGraph```，表示历史 episode 的节点/区域结构。
4. ```belief_manager.py```：```BeliefManager``` 和 ```WorldModel``` 管理问题、工作记忆与探索；```eqa_reasoning.py``` 的 ```EQAReasoning``` 决定准备回答、目标识别并生成最终答案。
5. ```mind_palace_exploration.py```：```MindPalaceExploration``` 汇集 ```EpisodicExploration.plan```、```RoomExploration.plan```、```PlaceExploration.plan```，并调用 ```OpenAIInterface``` 完成相关 LLM/VLM 推理；```WorldModel.explore``` 从指定 episode / viewpoint 获取观测。
6. ```eqa_evaluation.py```：```EQAEvaluation``` 评估 answer accuracy、SPL / 探索代价、检索图像数。notebook 结果是离线运行样例，不能代表本次复跑。

## 复现与开放边界

- **可用**：公开 Python / notebook 算法框架、问题标注、场景下载入口；参考实现依赖 OpenAI API，需配置 ```OPENAI_API_KEY```。
- **未核验为开源**：论文实景腿式机器人完整感知、路线执行、现场建图、部署配置与长期采集流水线；不要将离线 ```WorldModel.explore``` 说成已公开的端到端真机导航服务。
- **本次未运行**：6 GB 场景文件下载、全量 150 题评估、OpenAI API 调用及真实机器人测试。

## Wiki 映射

- [Mind Palace 论文实体](../../wiki/entities/paper-mind-palace-laeqa.md)
- [论文原文](../papers/mind_palace_la_eqa_arxiv_2507_12846.md)
- [项目页](../sites/mind-palace-laeqa.md)
