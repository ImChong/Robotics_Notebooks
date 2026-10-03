# LIBERO

> 来源归档（Humanoid Motion Intelligence 开源项目主表 + 官方 LIBERO 项目资料）

- **标题：** LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning
- **简称：** LIBERO
- **类型：** repo / robot-manipulation benchmark
- **技术路线分组：** 工程与实机部署（上游主表分类）
- **GitHub：** <https://github.com/Lifelong-Robot-Learning/LIBERO>
- **项目页：** <https://libero-project.github.io/>
- **文档：** <https://lifelong-robot-learning.github.io/LIBERO/>
- **官方数据页：** <https://libero-project.github.io/datasets>
- **Hugging Face 数据集：** <https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets>
- **论文：** [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310) — NeurIPS 2023 Datasets and Benchmarks Track
- **论文详情页：** [paper-rcl-2306-03310-libero-benchmarking-knowledge-transfer-for-lifel](../../wiki/entities/paper-rcl-2306-03310-libero-benchmarking-knowledge-transfer-for-lifel.md)
- **仿真环境：** [robosuite](https://github.com/ARISE-Initiative/robosuite)；当前 requirements 固定版本 1.4.0
- **安装说明：** [LIBERO installation](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html)
- **依赖清单：** [requirements.txt](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/requirements.txt)
- **入库日期：** 2026-07-30（项目入口）；2026-10-03 补齐论文、数据、仿真依赖链接
- **一句话说明：** LIBERO 以 130 个语言条件机器人操作任务和人类遥操作演示，评测终身学习中的知识迁移；任务变化覆盖空间关系、物体、目标及其组合。
- **开源状态：** 官方 GitHub 仓库提供任务基准、训练和评估代码及数据下载脚本；代码与数据许可分别以仓库 LICENSE 和官方说明为准。
- **策展入口：** [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)
- **相关论文来源：** [论文来源归档](../papers/rcl_awesome_wam_2306_03310_libero-benchmarking-knowledge-transfer-f.md)
- **沉淀到 wiki：** 是 → [libero-benchmark 工程页](../../wiki/entities/libero-benchmark.md)

## 项目资料说明

- 论文提出四个套件，共 130 个任务：LIBERO-Spatial、LIBERO-Object、LIBERO-Goal、LIBERO-100；后者在设置中分为 LIBERO-90 与 LIBERO-10。
- 官方 README 提供演示数据下载脚本，也支持从 Hugging Face 选择数据来源。Hugging Face 页面托管 HDF5 格式演示数据。
- 官方数据页列出工作区与腕部 RGB 图像、本体状态、语言任务说明和 PDDL 场景描述等数据内容。
- LIBERO 将 robosuite 用作底层仿真环境；其 requirements.txt 将 robosuite 固定为 1.4.0。升级仿真环境时应先检查基准任务与评估的兼容性。
- 版本、依赖、任务定义、训练和评估细节以官方仓库及论文为准。

## 对 wiki 的映射

- [LIBERO 项目与基准入口](../../wiki/entities/libero-benchmark.md)
- [LIBERO 论文详情页](../../wiki/entities/paper-rcl-2306-03310-libero-benchmarking-knowledge-transfer-for-lifel.md)
- [robosuite 仿真环境实体](../../wiki/entities/robosuite.md)
- [robosuite 论文实体](../../wiki/entities/paper-as-2009-12293-robosuite-a-modular-simulation-framework-and-ben.md)
- [论文原始来源归档](../papers/rcl_awesome_wam_2306_03310_libero-benchmarking-knowledge-transfer-f.md)
