---
type: source-site
updated: 2026-11-21
url: https://drake.mit.edu/
---

# Drake Official Overview & MIT Underactuated Appendix A

- **来源 URL：** <https://drake.mit.edu/> 与 <https://underactuated.csail.mit.edu/drake.html>
- **维护机构：** Toyota Research Institute (TRI) / MIT CSAIL Robot Locomotion Group（Russ Tedrake 等）
- **核心架构：** C++ 核心库（暴露解析梯度、稀疏结构与多项式/接触方程结构，区别于黑箱仿真器）+ `pydrake` Python 绑定 + `MathematicalProgram` 统一优化接口（对接 OSQP / SNOPT / IPOPT / MOSEK / Gurobi）。
- **三大支柱模块（官网首屏与 MIT App. A）：**
  1. `Systems` 框架（`Modeling Dynamical Systems`：连续/离散/混合状态机 `Diagram` 组合）。
  2. `MultibodyPlant` 与 `SceneGraph`（多体动力学、接触与摩擦仿真、空间几何查询）。
  3. `MathematicalProgram`（LP/QP/SOCP/SDP/NLP 抽象，直接转录与直接配点轨迹优化）。
- **工程注记（App. A）：** `pydrake` 适用于快速原型，自动生成的 Python API 文档较 C++ 滞后，官方建议查阅 C++ 文档定位类与方法后再核对 Python 绑定签名。
- **通读：** 已完成
- **核对范围：** 已通读 drake.mit.edu 首页（133 行）与 MIT Underactuated Robotics Ch. 22 / Appendix A（drake.html 116 行）及 Ch. 10 trajopt.html（Direct Transcription / Collocation / Shooting / MPC 章节）。
