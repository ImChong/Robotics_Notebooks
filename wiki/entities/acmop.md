---
type: entity
tags: [software, motor, optimization, femm, jmag, open-source, research, pmsm, acmop]
status: complete
updated: 2026-07-25
related:
  - ../comparisons/open-source-torque-motor-em-design.md
  - ../comparisons/motor-em-simulation-software.md
  - ./pyleecan.md
  - ./axfluxmdo.md
  - ./femm-foc-simulation.md
  - ../../roadmap/depth-torque-motor-design.md
sources:
  - ../../sources/repos/acmop.md
  - ../../sources/personal/open_source_torque_motor_em_design_curator.md
summary: "ACMOP：交流电机自动优化研究框架；几何参数化 + FEMM/JMAG + BH 曲线 + 多目标 Pareto 与报告；环境依赖偏旧，适合研究扫参优化，不建议作为力矩电机第一入门项目。"
---

# ACMOP（交流电机自动优化框架）

## 一句话定义

**ACMOP**（[horychen/ACMOP](https://github.com/horychen/ACMOP)，*Alternating Current Machine Optimization Project*）是一套 **交流电机几何参数化 + 有限元自动优化** 研究代码：对接 **FEMM** 与 **JMAG**，含材料 BH 曲线、多目标优化与设计报告生成。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ACMOP | Alternating Current Machine Optimization Project | 本项目名称 |
| FEMM | Finite Element Method Magnetics | 开源 2D 电磁有限元 |
| JMAG | JMAG Designer | 商业电机电磁仿真套件 |
| Pareto | Pareto front | 多目标非支配解集 |
| BH | B–H curve | 磁化曲线 |

## 为什么重要

- 展示「如何 **自动改** 槽宽、齿宽、磁钢厚度 → 跑 FEA → 比力矩/效率/损耗/脉动 → 出 Pareto」的完整研究工作流。
- 同时保留 **免费 FEMM** 与 **商业 JMAG** 两条求解路径，便于对照 [仿真软件选型](../comparisons/motor-em-simulation-software.md)。
- 与 [PYLEECAN](./pyleecan.md) / [axfluxmdo](./axfluxmdo.md) 同属工具层；ACMOP 更偏作者自用优化框架与轴承电机等研究案例。

## 核心原理

```mermaid
flowchart LR
  tmpl["初始/模板设计"]
  geo["参数化截面\nCrossSect*"]
  fea["FEMM_Solver / JMAG"]
  swarm["群体优化\nswarm_data 可重启"]
  report["性能报告 / Pareto"]
  tmpl --> geo --> fea --> swarm --> report
```

`codes3/` 含截面类、`FEMM_Solver.py`、`JMAG.py`、多种 PMSM/轴承电机问题定义；`BH/` 提供材料曲线。

## 工程实践

| 项 | 说明 |
|----|------|
| 何时用 | 已有 FEMM/JMAG 基础，要研究自动优化与约束处理 |
| 何时不用 | 第一次学关节电机——先 Ironless / FEMM-FOC / PYLEECAN |
| 环境 | README 指向偏旧 JMAG 17.1 与 Anaconda/Python；升级需自测 API |
| 许可 | 仓库无 SPDX；引用与再分发前自行确认 |

## 局限与风险

- **研究代码成熟度**：依赖旧工具链；约束与重启逻辑 README 自列多项 TODO。
- **不是人形硬件仓**：无面向髋膝的固定 CAD/BOM/台架。
- 磁钢涡流等效应在 FEMM 路径上仍有作者自述缺口。

## 源码运行时序图

入口：`python codes3/acmop.py`（文件底部 `main(n)`：`n=3` 单点评估，`n=4` 多目标优化，`n=31` 几何绘制；仓库默认分支 HEAD，见下方依据）
```mermaid
sequenceDiagram
    autonumber
    participant U as 用户 命令行
    participant W as acmop.py AC_Machine_Optiomization_Wrapper
    participant T as bearingless_spmsm_design 等模板函数
    participant D as acm_designer.acm_designer
    participant V as 各 *_design_variant 设计变体类
    participant F as JMAG.py 或 FEMM_Solver
    participant P as Problem_BearinglessSynchronousDesign pygmo UDP

    U->>W: 构造 Wrapper 传入 select_spec 与 FEA 配置
    W->>W: __post_init__ 载入 spec 与 fea_config 字典
    W->>T: part_initialDesign 按 select_spec 选择模板函数
    T-->>W: acm_template
    W->>D: 构造 acm_designer 传入 template
    U->>W: main(n) 分派到 part_evaluation 或 part_optimization
    alt n=3 单点评估
        W->>D: evaluate_design_json_wrapper(template, x_denorm)
    else n=4 多目标优化
        W->>P: 创建 UDP 与 pg.population
        P->>D: fitness(x) 调用 builtins.ad.evaluate_design_json_wrapper
    end
    D->>D: evaluate_design 按 machine_type 分派到 fea_wrapper
    D->>V: build_acm_variant 生成几何与设计变体
    V-->>D: acm_variant
    alt fea_config 含 JMAG
        D->>F: build_jmag_project 驱动 JMAG 求解
        F-->>D: build_str_results 读取 CSV 结果
    else fea_config 含 FEMM
        D->>F: build_femm_project 建 FEMM 工程并计算
        F-->>D: 性能结果
        D->>D: save_to_disk 写 spec_performance_dict
    end
    D-->>P: 目标值与约束值
    P-->>W: pygmo MOEA/D 迭代更新种群
```

图依据：horychen/ACMOP@745c241（默认分支 HEAD），实际读过 `README.md`、`codes3/acmop.py`、`codes3/acm_designer.py`、`codes3/Problem_BearinglessSynchronousDesign.py`；`codes3/population.py`、`codes3/FEMM_Solver.py`、`codes3/JMAG.py` 仅按函数名检索。未运行代码。

## 关联页面

- [开源力矩电机电磁设计完整度对比](../comparisons/open-source-torque-motor-em-design.md)
- [PYLEECAN](./pyleecan.md) · [axfluxmdo](./axfluxmdo.md) · [FEMM-FOC](./femm-foc-simulation.md)
- [电机电磁仿真软件选型](../comparisons/motor-em-simulation-software.md)

## 参考来源

- [sources/repos/acmop.md](../../sources/repos/acmop.md)
- [开源力矩电机电磁设计策展](../../sources/personal/open_source_torque_motor_em_design_curator.md)
- 源码核对（2026-11-02）：horychen/ACMOP@745c241（默认分支 HEAD），读过 README.md、codes3/acmop.py、codes3/acm_designer.py、codes3/Problem_BearinglessSynchronousDesign.py、test.py；codes3/population.py、codes3/FEMM_Solver.py、codes3/JMAG.py 仅按函数名检索；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- 仓库（分支 `better_framework`）：<https://github.com/horychen/ACMOP>
- FEMM 损耗参考（README）：<http://www.femm.info/wiki/SPMLoss>
