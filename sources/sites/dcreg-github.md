# DCReg GitHub 项目页

- **URL：** <https://github.com/JokerJohn/DCReg>
- **Wiki：** <https://github.com/JokerJohn/DCReg/wiki>
- **论文：** [dcreg_ijrr_2026_hu.md](../papers/dcreg_ijrr_2026_hu.md)
- **代码：** [dcreg.md](../repos/dcreg.md)

## 核查摘要（2026-09-21）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — `main` 完整 DCReg；`baseline` 为早期基线快照 |
| **构建** | `cmake -S DCReg -B DCReg/build`；C++17；Eigen3 + PCL |
| **可执行入口** | `dcreg_minimal_example`、`dcreg_runner`、`dcreg_parking_lot_example` |
| **数据** | 仓内小样本 + Drive 外链大 prior map |
| **上游** | Open3D #7482、PCL #6432 集成 PR 进行中 |
| **待发布** | README 计划 DCReg 定位系统整管线开源 |

## 一句话说明

IJRR 2026 DCReg 官方仓库：退化 LiDAR 配准三模块实现与停车场 demo，项目页即 GitHub README + Wiki。
