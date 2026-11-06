# Awesome AI CAE README 通读（CAE/CFD 代理工具总览视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/cae-cfd-agent-skills-landscape.md`）

- **标题：** Awesome AI-CAE — The CAE tools an AI agent can actually call
- **作者 / 维护者：** kimimgo（CC0-1.0）
- **类型：** repo / awesome-list
- **原文：** <https://raw.githubusercontent.com/kimimgo/awesome-ai-cae/main/README.md>（2026-11 经代理获取，约 424 行，含 2026-10-12 更新的评分表）
- **一句话说明：** 按"代理可调用性"而非 star 数给 110+ 个 CAE/CAD/网格/可视化/科学 ML 工具排序的策展清单，给出 0-100 的 AI-Readiness 评分规则。
- **核对范围：** 通读 README 全文：Quickstart、AI-Readiness Index（前 15 名）、评分方法与 Verified/Declared/Roadmap 说明、Core Engine Readiness 表、MCP Servers、CFD、FEA、SPH、DEM、可视化、CAD、网格、可微仿真、AI/ML、代理模型与 PINN、优化、数据格式、数据集、学习资源、贡献与许可。未读 `READINESS.md` 全表、`scripts/readiness-score.py`、`data/*.json` 与各被收录仓库自身的 README（各仓库另有各自归档，不在本次范围）；未获取微信公众号文章全文（本站仅有其摘要归档）。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位：** 只收可无头（headless）通过 MCP、Python 或 CLI 驱动的开源仿真/CAD/网格工具，"按可调用性排名，不按 star"；另有少量相邻的数据集与学习资源。
- **MCP 服务器仅 3 个：** viznoir（22 个渲染/切片/等值面/体渲染/动画工具，处理 OpenFOAM/VTK/CGNS）、llnl/paraview_mcp（自然语言控制 ParaView，多模态 LLM 观察视口反馈）、webworn/openfoam-mcp-server（带苏格拉底式提问的 CFD 教学与排错）。其余工具靠 Python/CLI 脚本化调用。
- **AI-Readiness 评分（0-100）：** MCP 35、Python API 25、CLI/REST 15、近 6 个月有维护 15、采用度（star，对数缩放）10；pip 可安装为额外 +15，总分封顶 100。分级：AI-Native 75+、Agent-Ready 50-74、Scriptable 30-49、Experimental 低于 30。2026-10-12 统计 114 个工具：2 AI-Native、61 Agent-Ready、28 Scriptable、23 Experimental。榜首为 paraview_mcp（95）与 viznoir（93），其后为 taichi、MuJoCo、rerun 等（65 分）。
- **Verified 与 Declared 的区分：** 已验证的有 GitHub star/活跃度、PyPI 可用性、安装并 import 的冒烟测试；MCP 与 CLI/API 标签仅为条目自报（declared），尚未回放端到端代理调用；路线图是做 MCP 握手与无头运行的执行验证及"代理调用记录"。
- **核心求解器编辑表（17 个）：** OpenFOAM、FEniCS(dolfinx)、Gmsh、VTK/ParaView、SU2、MFEM、deal.II、DualSPHysics、Taichi、PyFR、CalculiX、Elmer、OpenCASCADE、MOOSE、FreeFEM、SfePy、MuJoCo；列为 Python 绑定、Headless、Docker、AI-Native，属维护者主观判断。
- **分类覆盖：** CFD（OpenFOAM、SU2、Nek5000/nekRS、preCICE、FluidX3D、PyFR）、FEA（CalculiX、deal.II、Elmer、dolfinx、Firedrake、MOOSE、Kratos、MFEM、OpenSees 等）、SPH、DEM、可视化、CAD（CadQuery、build123d、FreeCAD、OCCT）、网格（Gmsh、netgen、TetGen、fTetWild、MeshAnything 等）、可微仿真（XLB、Brax、Warp、Taichi、JAX-Fluids、MuJoCo）、AI/ML（Foam-Agent、MetaOpenFOAM、jax-cfd、neuraloperator、PhysicsNeMo）、代理模型/PINN、优化（BoTorch、OpenMDAO、pymoo）、数据格式（meshio、CGNS、HDF5）、数据集（AirfRANS、DrivAerNet、PDEBench、NASA TMR 等）、学习资源。
- **与本站总览的关系：** 本站总览的 53 个仓库页主要来自微信文章的商用软件 MCP 与技能仓库，而该 README 偏重开源求解器与 Python 库；二者重合度有限，README 中的 MCP 只有 viznoir、ParaView MCP、OpenFOAM MCP 三个出现在本站总览表中。

## 对 wiki 的映射

- [`wiki/overview/cae-cfd-agent-skills-landscape.md`](../../wiki/overview/cae-cfd-agent-skills-landscape.md)
- [`wiki/entities/kimimgo-awesome-ai-cae.md`](../../wiki/entities/kimimgo-awesome-ai-cae.md)
