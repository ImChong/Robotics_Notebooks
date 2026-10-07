---
type: overview
tags:
  - engineering-tools
  - cae
  - cfd
status: complete
updated: 2026-10-07
summary: "按项目身份整理 CAE/CFD 代理技能、MCP、CLI 和可视化仓库；每个外部仓库都有独立详情节点，避免同名项目混在一页。"
sources:
  - ../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md
---

# CAE / CFD 代理工具与技能项目总览

## 一句话定义

本页将一篇 CAE/CFD 工程代理工具清单中的公开仓库拆成独立、可追溯的项目详情节点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CAE | Computer-Aided Engineering | 计算机辅助工程工具和分析流程 |
| CFD | Computational Fluid Dynamics | 计算流体力学求解与后处理 |
| MCP | Model Context Protocol | 代理发现并调用外部软件工具的协议 |
| FEA | Finite Element Analysis | 有限元分析工作流 |

## 为什么重要

工程代理项目常把技能说明、工具连接、求解器执行和结果审核混在一起。按仓库建立独立节点，可以分别追踪它提供的是流程指导、执行接口还是软件编排；同名仓库也不会因名称相同而丢失身份差异。

## 核心结构

文章条目可按以下工程角色阅读：

- **资源目录与技能集合：** 用来发现工具或分发可复用流程，通常不替代求解器。
- **执行连接层：** MCP、CLI 或插件将代理请求传递到专业软件，受软件安装、版本及许可证约束。
- **验证与证据层：** 网格独立性、契约、残差/守恒量检查和可复现输出，用来判断自动化步骤是否有据可查。
- **专用工作流：** 设计优化、CAD 建模、科研写作等特定任务闭环。

## 工程实践

先识别仓库的实际角色，再检查默认分支 README、许可证、安装依赖和可运行示例。对仿真工作流，应以网格、边界、守恒和物理结果作为验证证据；对 MCP 执行接口，确认调用权限和任意代码执行边界；对商业软件确认本机安装与有效授权。Skill 文件可以规划和检查流程，但其本身不提供求解器能力。

## 项目详情节点

| 类别 | 独立详情节点 | 官方仓库 |
|------|--------------|----------|
| 目录与工具发现 | [Awesome AI CAE](../entities/kimimgo-awesome-ai-cae.md) | [kimimgo/awesome-ai-cae](https://github.com/kimimgo/awesome-ai-cae) |
| 目录与工具发现 | [HPC-Skills](../entities/scimate-ai-hpc-skills.md) | [SciMate-AI/HPC-Skills](https://github.com/SciMate-AI/HPC-Skills) |
| 目录与工具发现 | [CAE Agent Hub](../entities/cai-aa-cae-agent-hub.md) | [Cai-aa/CAE-Agent-Hub](https://github.com/Cai-aa/CAE-Agent-Hub) |
| 目录与工具发现 | [Claude Engineering Skills](../entities/soljourner-claude-engineering-skills.md) | [Soljourner/claude-engineering-skills](https://github.com/Soljourner/claude-engineering-skills) |
| CFD 与 OpenFOAM | [OpenFOAM Claude Suite](../entities/swtbkim-openfoam-claude-suite.md) | [swtbkim/openfoam-claude-suite](https://github.com/swtbkim/openfoam-claude-suite) |
| CFD 与 OpenFOAM | [OpenFOAM Expert Skill](../entities/zyzhan417-openfoam-expert-skill.md) | [Zyzhan417/OpenFOAM_expert_SKILL](https://github.com/Zyzhan417/OpenFOAM_expert_SKILL) |
| CFD 与 OpenFOAM | [OpenFOAM Simulation](../entities/ezrajay2333-openfoam-simulation.md) | [EzraJay2333/openfoam-simulation](https://github.com/EzraJay2333/openfoam-simulation) |
| CFD 与 OpenFOAM | [Fluent CFD Skill](../entities/cavoiie-fluent-cfd-skill.md) | [cavoiie/fluent-cfd-skill](https://github.com/cavoiie/fluent-cfd-skill) |
| CFD 与 OpenFOAM | [AutoStar](../entities/ouscar-ou-autostar.md) | [Ouscar-ou/AutoStar](https://github.com/Ouscar-ou/AutoStar) |
| CFD 与 OpenFOAM | [OpenFOAM CFD Codex Skill](../entities/hnuvv-openfoam-cfd-codexskill.md) | [HNUVV/openfoam-CFD-codexskill](https://github.com/HNUVV/openfoam-CFD-codexskill) |
| CFD 与 OpenFOAM | [OpenFOAM Agent Skills](../entities/twj011-openfoam-agent-skills.md) | [twj011/openfoam-agent-skills](https://github.com/twj011/openfoam-agent-skills) |
| CFD 与 OpenFOAM | [OpenFOAM MCP Server](../entities/webworn-openfoam-mcp-server.md) | [webworn/openfoam-mcp-server](https://github.com/webworn/openfoam-mcp-server) |
| CFD 与 OpenFOAM | [Foam-Agent](../entities/csml-rpi-foam-agent.md) | [csml-rpi/Foam-Agent](https://github.com/csml-rpi/Foam-Agent) |
| 结构 FEA | [Abaqus Agent Skills](../entities/1348109517-abaqus-agent-skills.md) | [1348109517/abaqus-agent-skills](https://github.com/1348109517/abaqus-agent-skills) |
| 结构 FEA | [FEMIS Skill](../entities/test1card-femis-skill.md) | [test1card/femis-skill](https://github.com/test1card/femis-skill) |
| 结构 FEA | [FEP Agent Hub](../entities/s2mon123-fep-agent-hub.md) | [S2mon123/FEP-Agent-Hub](https://github.com/S2mon123/FEP-Agent-Hub) |
| 结构 FEA | [Abaqus Agent](../entities/nellikassa566-ops-abaqus-agent.md) | [nellikassa566-ops/abaqus-agent](https://github.com/nellikassa566-ops/abaqus-agent) |
| 结构 FEA | [Agentic CAE](../entities/wogokoro-agentic-cae.md) | [wogokoro/Agentic-CAE](https://github.com/wogokoro/Agentic-CAE) |
| 结构 FEA | [CAE Physics Simulations Agent](../entities/dipanbartaula-cae-physics-simulations-agent.md) | [DipanBartaula/CAE_Physics_Simulations_Agent](https://github.com/DipanBartaula/CAE_Physics_Simulations_Agent) |
| 商用软件 MCP | [COMSOL Multiphysics MCP](../entities/wjc9011-comsol-multiphysics-mcp.md) | [wjc9011/COMSOL_Multiphysics_MCP](https://github.com/wjc9011/COMSOL_Multiphysics_MCP) |
| 商用软件 MCP | [COMSOL MCP](../entities/777gegewu-comsol-mcp.md) | [777gegewu/comsol-mcp](https://github.com/777gegewu/comsol-mcp) |
| 商用软件 MCP | [Ansys MCP Server（PyAnsys）](../entities/vorobjewsen30-max-ansys-mcp-server.md) | [vorobjewsen30-max/ansys-mcp-server](https://github.com/vorobjewsen30-max/ansys-mcp-server) |
| 商用软件 MCP | [Ansys MCP Server（多产品版）](../entities/knewnothing-git-ansys-mcp-server.md) | [knewnothing-git/ansys-mcp-server](https://github.com/knewnothing-git/ansys-mcp-server) |
| 商用软件 MCP | [ANSYS Workbench MCP](../entities/hongwenwang36-eng-ansys-workbench-mcp.md) | [hongwenwang36-eng/ANSYS-Workbench-mcp](https://github.com/hongwenwang36-eng/ANSYS-Workbench-mcp) |
| 商用软件 MCP | [AnsysAgent](../entities/sduwby-ansysagent.md) | [sduwby/AnsysAgent](https://github.com/sduwby/AnsysAgent) |
| 商用软件 MCP | [sim-cli](../entities/svd-ai-lab-sim-cli.md) | [svd-ai-lab/sim-cli](https://github.com/svd-ai-lab/sim-cli) |
| 商用软件 MCP | [sim-plugin-openfoam](../entities/svd-ai-lab-sim-plugin-openfoam.md) | [svd-ai-lab/sim-plugin-openfoam](https://github.com/svd-ai-lab/sim-plugin-openfoam) |
| 商用软件 MCP | [sim-plugin-starccm](../entities/svd-ai-lab-sim-plugin-starccm.md) | [svd-ai-lab/sim-plugin-starccm](https://github.com/svd-ai-lab/sim-plugin-starccm) |
| CAD、网格与可视化 | [ParaView MCP](../entities/llnl-paraview-mcp.md) | [llnl/paraview_mcp](https://github.com/llnl/paraview_mcp) |
| CAD、网格与可视化 | [viznoir](../entities/kimimgo-viznoir.md) | [kimimgo/viznoir](https://github.com/kimimgo/viznoir) |
| CAD、网格与可视化 | [CAD CAE Copilot](../entities/armpro24-blip-cad-cae-copilot.md) | [armpro24-blip/cad-cae-copilot](https://github.com/armpro24-blip/cad-cae-copilot) |
| CAD、网格与可视化 | [FreeCAD Automation Skill（Cai-aa）](../entities/cai-aa-freecad-automation-skill.md) | [Cai-aa/freecad-automation-skill](https://github.com/Cai-aa/freecad-automation-skill) |
| CAD、网格与可视化 | [FreeCAD Automation Skill（miaooo0000OOOO）](../entities/miaooo0000oooo-freecad-automation-skill.md) | [miaooo0000OOOO/freecad-automation-skill](https://github.com/miaooo0000OOOO/freecad-automation-skill) |
| CAD、网格与可视化 | [FreeCAD AI Skill（reyk）](../entities/reyk-freecad-ai-skill.md) | [reyk/freecad-ai-skill](https://github.com/reyk/freecad-ai-skill) |
| CAD、网格与可视化 | [FreeCAD AI Skill（shanputaoye）](../entities/shanputaoye-freecad-ai-skill.md) | [shanputaoye/freecad-ai-skill](https://github.com/shanputaoye/freecad-ai-skill) |
| CAD、网格与可视化 | [FreeCAD Engineering](../entities/v0v1kkk-freecad-engineering.md) | [V0v1kkk/freecad-engineering](https://github.com/V0v1kkk/freecad-engineering) |
| CAD、网格与可视化 | [AnkusDrive](../entities/gchen19-ankusdrive.md) | [gchen19/AnkusDrive](https://github.com/gchen19/AnkusDrive) |
| CAD、网格与可视化 | [FreeCAD MCP（sandraschi）](../entities/sandraschi-freecad-mcp.md) | [sandraschi/freecad-mcp](https://github.com/sandraschi/freecad-mcp) |
| CAD、网格与可视化 | [FreeCAD MCP（Tessalabs）](../entities/tessalabs-space-freecad-mcp.md) | [tessalabs-space/freecad-mcp](https://github.com/tessalabs-space/freecad-mcp) |
| CAD、网格与可视化 | [SolidWorks Automation Skill](../entities/wzyn20051216-solidworks-automation-skill.md) | [wzyn20051216/solidworks-automation-skill](https://github.com/wzyn20051216/solidworks-automation-skill) |
| CAD、网格与可视化 | [CAD Operations Skill](../entities/2836048681-cad-operations-skill.md) | [2836048681/cad-operations-skill](https://github.com/2836048681/cad-operations-skill) |
| 专项设计与研究 | [AeroDesign Skill](../entities/pan-chenliang-aerodesign-skill.md) | [Pan-Chenliang/AeroDesign_skill](https://github.com/Pan-Chenliang/AeroDesign_skill) |
| 专项设计与研究 | [Vortex Funnel Generator](../entities/hooyao-vortex-funnel-gen.md) | [hooyao/vortex-funnel-gen](https://github.com/hooyao/vortex-funnel-gen) |
| 专项设计与研究 | [CFD SciPaper Agent](../entities/xiuyiw-cfd-scipaper-agent.md) | [Xiuyiw/CFD-SCIPaper-Agent](https://github.com/Xiuyiw/CFD-SCIPaper-Agent) |
| 专项设计与研究 | [Claude Skills for Computational Designers](../entities/abhinavbwj-claude-skills-for-computational-designers.md) | [Abhinavbwj/Claude-skills-for-Computational-Designers](https://github.com/Abhinavbwj/Claude-skills-for-Computational-Designers) |
| 专项设计与研究 | [SCI Mech Fluid Polishing](../entities/flyingakai-sci-mech-fluid-polishing.md) | [FlyingAkai/sci-mech-fluid-polishing](https://github.com/FlyingAkai/sci-mech-fluid-polishing) |
| 专项设计与研究 | [Scientific Agents](../entities/k-dense-ai-scientific-agents.md) | [K-Dense-AI/scientific-agents](https://github.com/K-Dense-AI/scientific-agents) |
| 商用软件 MCP | [Mechanical MCP（PyMechanical gRPC）](../entities/codersag-mechanical-mcp.md) | [codersag/mechanical-mcp](https://github.com/codersag/mechanical-mcp) |
| 商用软件 MCP | [Ansys AEDT MCP](../entities/laplaceyoung-ansys-aedt-mcp.md) | [LaplaceYoung/ansys-aedt-mcp](https://github.com/LaplaceYoung/ansys-aedt-mcp) |
| 商用软件 MCP | [HFSS MCP Server](../entities/gfgf2023-hfss-mcp-server.md) | [gfgf2023/hfss-mcp-server](https://github.com/gfgf2023/hfss-mcp-server) |
| 商用软件 MCP | [Lumerical FDTD MCP](../entities/leisymqaz-lumerical-fdtd-mcp.md) | [leisymqaz/lumerical-fdtd-mcp](https://github.com/leisymqaz/lumerical-fdtd-mcp) |
| 商用软件 MCP | [STK MCP](../entities/alti3-stk-mcp.md) | [alti3/stk-mcp](https://github.com/alti3/stk-mcp) |
| 商用软件 MCP | [CONVERGE Studio MCP](../entities/legiiiit-converge-studio-mcp.md) | [Legiiiit/converge-studio-mcp](https://github.com/Legiiiit/converge-studio-mcp) |


## 局限与风险

此总览包含 53 个独立公开仓库节点，来源是一篇策展文章及对应的公开仓库资料，不构成工具性能排名或工程认证。文章另提到“ANSYS-automatic-wwj”，但没有可核实的 owner/仓库链接，暂不建身份未明的项目页。同名项目需按仓库 owner/repo 辨别。文章未给出完整仓库地址的少数项目，只有在公开仓库身份能核实时才映射；映射依据在详情页标注。任何 Skill/MCP 输出都需用户检查工程假设和软件许可。

## 关联页面

- [OpenFOAM Claude Suite](../entities/swtbkim-openfoam-claude-suite.md) — 代表性的 OpenFOAM 分阶段工作流
- [FEMIS Skill](../entities/test1card-femis-skill.md) — 仿真验证与证据治理
- [CAE Agent Hub](../entities/cai-aa-cae-agent-hub.md) — 多软件资源与执行层拆分

## 参考来源

- [微信公众号原文归档](../../sources/blogs/wechat_cae_cfd_agent_skills_2026-10-07.md)
- [完整仓库资料与源码开放状态](../../sources/repos/kimimgo-awesome-ai-cae.md)

## 推荐继续阅读

- [文章原文](https://mp.weixin.qq.com/s/8bv-ySeocRQRh1HRT5S3SA) — 作者对 CAE/CFD Agent Skills 和 MCP 项目的分类及使用提醒
