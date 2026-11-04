---
type: entity
project_id: hooyao-vortex-funnel-gen
code: "https://github.com/hooyao/vortex-funnel-gen"
tags:
  - engineering-tools
  - caa-cfd
  - engineering
status: complete
updated: 2026-10-07
summary: "Vortex Funnel Generator（hooyao/vortex-funnel-gen）是用 AI 设计 3D 打印防咕咚涡流漏斗，贯通 CadQuery 参数建模、OpenFOAM interFoam VOF 多相计算、Docker 执行和多保真度贝叶斯优化"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
  - ../../sources/repos/hooyao-vortex-funnel-gen.md
---

# Vortex Funnel Generator（hooyao/vortex-funnel-gen）

## 一句话定义

Vortex Funnel Generator（hooyao/vortex-funnel-gen）是用 AI 设计 3D 打印防咕咚涡流漏斗，贯通 CadQuery 参数建模、OpenFOAM interFoam VOF 多相计算、Docker 执行和多保真度贝叶斯优化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CFD | Computational Fluid Dynamics | 计算流体力学证据或流程 |
| CAD | Computer-Aided Design | 参数化工程几何设计 |
| AI | Artificial Intelligence | 辅助设计或研究的人工智能能力 |

## 为什么重要

对 CAE 自动化而言，代理能否稳定完成任务，取决于工程步骤是否清楚、工具边界是否明确、结果是否保留可审查证据。用 AI 设计 3D 打印防咕咚涡流漏斗，贯通 CadQuery 参数建模、OpenFOAM interFoam VOF 多相计算、Docker 执行和多保真度贝叶斯优化。

## 核心原理

**项目自身范围：** AI-driven design optimization loop for a 3D-printable anti-glugging funnel.

**工作流：** 参数化几何生成后运行 CFD，以仿真指标驱动低/高保真优化迭代，再输出可打印方案。

## 工程实践

1. 从目标问题识别所需的 Skill、MCP、CLI 或软件接口，不把指导文件当成求解器。
2. 依照仓库 README 检查安装前提、软件/许可证、输入文件和输出证据。
3. 小规模运行并逐项核对几何、网格、边界、求解状态和物理量，再决定是否扩大任务。

文章所述的项目特征：用 AI 设计 3D 打印防咕咚涡流漏斗，贯通 CadQuery 参数建模、OpenFOAM interFoam VOF 多相计算、Docker 执行和多保真度贝叶斯优化。

## 局限与风险

优化质量取决于多相模型、网格和目标函数；仿真得分需用实物试验校验，容器/液体条件也会影响结果。

源码状态核验：该 GitHub 仓库于 2026-10-07 可公开访问。仓库 API 报告许可证为 MIT。 

## 源码运行时序图

入口：`python optimization/loop.py --coarse 20 --validate 5 --refine 5 --nprocs 24 --output output/optimisation`（README「Reproduce without Claude Code」；需 Docker 镜像 `opencfd/openfoam-default:2406`；加 `--resume` 可续跑）

```mermaid
sequenceDiagram
    autonumber
    participant Loop as optimization/loop.py run_optimisation()
    participant Opt as skopt Optimizer
    participant Eval as loop.py evaluate()
    participant Geo as geometry/funnel_generator.py generate()（CadQuery）
    participant Run as cfd/runner.py run_case()
    participant Dk as Docker（OpenFOAM 容器）
    participant Ext as runner.py extract_fitness（pyvista）

    Loop->>Opt: Optimizer(dimensions=DIMENSIONS, n_initial_points)
    opt --resume
        Loop->>Opt: _load_history() 后逐条 tell(x, -reward)
    end

    loop Stage 1：粗网格，iteration < n_coarse
        Loop->>Opt: ask() 得到参数向量 x
        Loop->>Eval: evaluate(x, iteration, output_root, n_procs, "coarse")
        Eval->>Eval: params_from_vector(x)，check_constraints()（不可行则罚分返回）
        Eval->>Geo: generate(params, iter_dir)，validate_params 后建模并导出 STEP 与 STL
        Eval->>Run: run_case(stl, case_dir, n_procs, mesh_level)
        Run->>Dk: setup_case：复制 base_case 模板并注入 STL（docker run）
        Run->>Dk: run_mesh：surfaceTransformPoints → surfaceFeatureExtract → blockMesh → snappyHexMesh -overwrite
        Run->>Dk: run_solver：setFields → decomposePar → mpirun interFoam -parallel → reconstructPar
        Run->>Ext: extract_fitness(case_dir)，取喉部指标
        Ext-->>Run: air_core_diameter / throat_velocity / air_fraction_throat
        Run-->>Eval: cfd 结果
        Eval->>Eval: compute_reward(cfd)，写 iter_NNN 结果
        Eval-->>Loop: result（reward）
        Loop->>Opt: tell(x, -reward)
    end

    loop Stage 2：对粗网格可行解中排名前 n_fine_validate 的参数用细网格复算
        Loop->>Eval: evaluate(x, iteration, ..., "fine")
        Loop->>Opt: tell(x, -reward)
    end

    loop Stage 3：细网格精修，iteration < stage3_end
        Loop->>Opt: ask()
        Loop->>Eval: evaluate(x, iteration, ..., "fine")
        Loop->>Opt: tell(x, -reward)
    end

    Loop->>Loop: 选取 best_reward，写 history 与 summary.json
```

图依据：hooyao/vortex-funnel-gen@6933ecf，实际读过 `optimization/loop.py`（`evaluate`、`run_optimisation`、`main`）、`cfd/runner.py`（`run_case`、`setup_case`、`run_mesh`、`run_solver`、`extract_fitness` 的调用链与命令字符串）、`geometry/funnel_generator.py`（`generate` 入口与导出）、`README.md`；`compute_reward` 公式与 `validate_params` 的细节、OpenFOAM 运行结果未逐一核对，未运行代码。

## 关联页面

- [CAE / CFD 代理工具项目总览](../overview/cae-cfd-agent-skills-landscape.md)
- [OpenFOAM Simulation](./ezrajay2333-openfoam-simulation.md)

## 参考来源

- [公众号文章原始资料归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [GitHub 仓库资料归档](../../sources/repos/hooyao-vortex-funnel-gen.md)
- [hooyao/vortex-funnel-gen 官方仓库](<https://github.com/hooyao/vortex-funnel-gen>)

## 推荐继续阅读

- [项目 README](<https://github.com/hooyao/vortex-funnel-gen/blob/main/README.md>)