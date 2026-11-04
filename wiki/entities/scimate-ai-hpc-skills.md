---
type: entity
project_id: scimate-ai-hpc-skills
code: "https://github.com/SciMate-AI/HPC-Skills"
tags:
  - engineering-tools
  - caa-cfd
  - engineering
status: complete
updated: 2026-10-07
summary: "HPC-Skills（SciMate-AI/HPC-Skills）是覆盖 OpenFOAM、SU2、LS-DYNA、FEniCS、CalculiX、ElmerFEM、LAMMPS、GROMACS、VASP、ParaView、Gmsh 等二十余个求解器/工具，并延伸到 MPI、GPU、Spack、集群调度"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
  - ../../sources/repos/scimate-ai-hpc-skills.md
---

# HPC-Skills（SciMate-AI/HPC-Skills）

## 一句话定义

HPC-Skills（SciMate-AI/HPC-Skills）是覆盖 OpenFOAM、SU2、LS-DYNA、FEniCS、CalculiX、ElmerFEM、LAMMPS、GROMACS、VASP、ParaView、Gmsh 等二十余个求解器/工具，并延伸到 MPI、GPU、Spack、集群调度。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AI | Artificial Intelligence | 以代理调用工程工具的人工智能能力 |
| CAE | Computer-Aided Engineering | 计算机辅助工程工具与工作流 |
| API | Application Programming Interface | 软件之间调用功能的接口 |

## 为什么重要

对 CAE 自动化而言，代理能否稳定完成任务，取决于工程步骤是否清楚、工具边界是否明确、结果是否保留可审查证据。覆盖 OpenFOAM、SU2、LS-DYNA、FEniCS、CalculiX、ElmerFEM、LAMMPS、GROMACS、VASP、ParaView、Gmsh 等二十余个求解器/工具，并延伸到 MPI、GPU、Spack、集群调度。

## 核心原理

**项目自身范围：** Portable agent skills for HPC workflows across solvers and cluster tooling.

**工作流：** 标准目录 `skills/<name>/SKILL.md` 将具体软件说明与执行流程拆开，可作为 Codex 或 Claude Code 的代理指引；文章点名的 hpc-ls-dyna、hpc-calculix、hpc-fenics、hpc-elmerfem 都是同一仓库内技能，不另立项目节点。

## 工程实践

1. 从目标问题识别所需的 Skill、MCP、CLI 或软件接口，不把指导文件当成求解器。
2. 依照仓库 README 检查安装前提、软件/许可证、输入文件和输出证据。
3. 小规模运行并逐项核对几何、网格、边界、求解状态和物理量，再决定是否扩大任务。

文章所述的项目特征：覆盖 OpenFOAM、SU2、LS-DYNA、FEniCS、CalculiX、ElmerFEM、LAMMPS、GROMACS、VASP、ParaView、Gmsh 等二十余个求解器/工具，并延伸到 MPI、GPU、Spack、集群调度。

## 局限与风险

SKILL.md 只提供代理指引；实际计算仍要求本机/集群具备对应求解器、依赖、资源与授权。

源码状态核验：该 GitHub 仓库于 2026-10-07 可公开访问。仓库 API 报告许可证为 MIT。 

## 源码运行时序图

本仓库以 `skills/<name>/SKILL.md` 与 `references/` 文档为主，可执行代码集中在 `skills/hpc-orchestration/scripts/` 的 4 个 Python 工具。下图只画其中的作业生成与提交工具 `hpc_job_submitter.py`，不代表整个技能集合的运行时。

入口：`python3 skills/hpc-orchestration/scripts/hpc_job_submitter.py --command "<求解命令>" [--scheduler auto|slurm|pbs|lsf] [--submit]`。

```mermaid
sequenceDiagram
    autonumber
    participant A as "调用者<br/>AI Agent 或用户"
    participant Main as "main()<br/>hpc_job_submitter.py"
    participant Det as "detect_scheduler()<br/>hpc_job_submitter.py"
    participant Gen as "generate_slurm_script / generate_pbs_script / generate_lsf_script"
    participant FS as "本地文件<br/>script_name"
    participant Sch as "sbatch / qsub / bsub<br/>调度器命令"
    A->>Main: 运行命令并传入 --command 与资源参数
    Main->>Main: argparse 解析参数
    alt --scheduler 为 auto
        Main->>Det: detect_scheduler()
        Det->>Det: 依次尝试 sbatch --version, qsub --version, bsub -V
        Det-->>Main: slurm, pbs, lsf 或 unknown
        alt 结果为 unknown
            Main->>Main: 警告后默认使用 slurm
        end
    end
    Main->>Gen: 按调度器生成 #SBATCH, #PBS 或 #BSUB 头部
    Gen-->>Main: header 字符串
    Main->>Main: 拼接 module load, --command 与 AGENT-LOG 行
    Main->>FS: 写入 script_name
    alt 写入失败
        Main-->>A: 错误信息并退出码 1
    else 写入成功
        opt 指定了 --submit
            Main->>Sch: 调用 sbatch 或 qsub 传入脚本, bsub 通过标准输入
            alt 提交失败或命令不存在
                Sch-->>Main: 错误返回码
                Main-->>A: 错误信息并退出码 1
            else 提交成功
                Sch-->>Main: 作业 ID 等标准输出
                Main-->>A: 打印提交结果
            end
        end
    end
```

图依据：SciMate-AI/HPC-Skills@4ea8069，读过 `README.md`、`skills-index.md`（开头部分）、`skills/hpc-orchestration/SKILL.md`（开头部分）、`skills/hpc-orchestration/references/tools-and-scripts.md`、`skills/hpc-orchestration/scripts/hpc_job_submitter.py`（全文）；`hpc_job_monitor.py`、`hpc_log_tracker.py`、`hpc_slurm_deploy.py` 仅核对了入口与参数，未画入图中。未运行代码。

## 关联页面

- [CAE / CFD 代理工具项目总览](../overview/cae-cfd-agent-skills-landscape.md)
- [Awesome AI CAE](./kimimgo-awesome-ai-cae.md)

## 参考来源

- [公众号文章原始资料归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [GitHub 仓库资料归档](../../sources/repos/scimate-ai-hpc-skills.md)
- [SciMate-AI/HPC-Skills 官方仓库](<https://github.com/SciMate-AI/HPC-Skills>)
- 源码核对（2026-11-02）：SciMate-AI/HPC-Skills@4ea8069，读过 README.md、skills/hpc-orchestration/SKILL.md、skills/hpc-orchestration/references/tools-and-scripts.md、skills/hpc-orchestration/scripts/hpc_job_submitter.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [项目 README](<https://github.com/SciMate-AI/HPC-Skills/blob/main/README.md>)