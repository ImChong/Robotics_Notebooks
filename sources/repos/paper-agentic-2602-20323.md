# PhysMem 官方代码仓库归档

- **论文实体：** [PhysMem: Scaling Test-Time Memory for Embodied Physical Reasoning](../../wiki/entities/paper-agentic-2602-20323.md)
- **论文归档：** [paper-agentic-2602-20323.md](../papers/paper-agentic-2602-20323.md)
- **官方代码仓库：** https://github.com/haoyangli16/PhysMem （默认分支 `main`，README 声明 MIT License；截至核对日文件树未单列 `LICENSE` 文件）
- **官方项目主页：** https://phys-mem.github.io/
- **开源状态：** 核心框架已开源（包含三层记忆数据结构、共鸣门控、聚类巩固、假设验证与升格主循环、多后端 LLM 适配器及 `quickstart` / `reflect_vlm` 示例；三项真机硬件任务的底层运动学标定与完整提示词实例未随论文附录或当前仓库发布）
- **核对边界：** 静态代码与 README / 文件树核对，未在本地执行安装或运行实验。

## 仓库结构与已核对入口（`haoyangli16/PhysMem`）

```text
PhysMem/
├── README.md                              # 安装、三层记忆架构、Quick Start 与配置说明
├── pyproject.toml                         # 包元数据与可选依赖 extras（openai/gemini/qwen/faiss/clustering/all）
├── setup.py                               # 构建脚本
├── assets/
│   └── pipeline.jpg                       # 流程架构图
└── physmem/
    ├── __init__.py                        # 公共入口：导出 PhysMem, ScientificLearningConfig, Experience, Principle 等
    ├── core/                              # 核心记忆数据结构与检索写回
    │   ├── experience.py                  # Experience, MemoryBank（情景记忆）
    │   ├── hypothesis.py                  # Hypothesis, HypothesisStore, ExperienceCluster（工作记忆）
    │   ├── principle.py                   # Principle, PrincipleStore, PrincipleType（长期记忆：AVOID/PREFER/SEQUENCE/COMPARE）
    │   ├── index.py                       # FAISS / NumPy 向量相似度索引
    │   ├── retriever.py                   # 结合符号状态过滤的 k-NN 检索器
    │   └── writeback.py                   # 写回策略与记忆折叠/垃圾回收
    ├── learning/                          # 科学学习闭环引擎
    │   ├── consolidation.py               # 经验聚类 + 候选假设生成
    │   ├── verification.py                # 动作级归因检验 + 升格/证伪逻辑
    │   └── loop.py                        # ScientificLearningLoop（主调度循环）
    ├── llm/                               # 模型推理后端抽象
    │   ├── base.py                        # 抽象基类 BaseLLM
    │   └── providers.py                   # OpenAI / Kimi、Gemini、Qwen 适配器实现
    └── examples/                          # 运行示例
        ├── quickstart.py                  # 独立最小闭环演示（含原则感知策略学习曲线）
        └── reflect_vlm/
            ├── README.md                  # Reflect-VLM 积木插装集成说明
            └── run_with_physmem.py        # Reflect-VLM 仿真基准集成运行入口
```

- **典型安装与运行接口（README 核对）：**
  - 安装：`pip install .`（仅 NumPy 核心）或 `pip install ".[all]"`（含 LLM 提供商、FAISS 与聚类嵌入依赖）。
  - 核心集成 API：`create_llm(...)` + `PhysMem(config=ScientificLearningConfig(...), llm=llm)`；在交互循环中调用 `mem.get_active_knowledge_prompt()` 注入原则与活跃假设，执行动作后调用 `mem.record_experience(action=..., success=..., fail=..., fail_tag=..., symbolic_state=..., state_vec=...)` 与 `mem.end_episode(success=...)`，并支持无 LLM 规则模式与 `save_state()` / `load_state()` 持久化。
  - 仿真示例入口：`physmem/examples/quickstart.py`（自包含快速入门）与 `physmem/examples/reflect_vlm/run_with_physmem.py`（Reflect-VLM 积木装配任务集成）。
