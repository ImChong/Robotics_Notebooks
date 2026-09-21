# DCReg（JokerJohn/DCReg）

- **URL：** <https://github.com/JokerJohn/DCReg>
- **论文：** [DCReg: Decoupled Characterization for Efficient Degenerate LiDAR Registration](../papers/dcreg_ijrr_2026_hu.md)（IJRR 2026）
- **arXiv：** <https://arxiv.org/abs/2509.06285>
- **演示视频：** <https://www.bilibili.com/video/BV1jsHQzCEra/>

## 一句话说明

退化感知 LiDAR point-to-plane 配准：Schur 补谱检测 → 物理轴退化表征 → 结构化预条件 PCG；Eigen + PCL 轻量栈。

## 主要入口（README）

| 路径 / 目标 | 作用 |
|-------------|------|
| `DCReg/include/dcreg.hpp` | 三模块 API：`DegeneracyDetection` / `DegeneracyCharacterization` / PCG 求解 |
| `DCReg/build/dcreg_minimal_example` | 合成线性系统演示三模块（集成入口） |
| `DCReg/build/dcreg_runner` | 合成配准管线 + 四参数化对比 |
| `DCReg/build/dcreg_parking_lot_example` | 真实停车场 scan-to-map 单帧案例 |
| `scripts/visualize_parking_lot_example.py` | 可选 Open3D 可视化（Python，非 C++ 核心依赖） |

## 构建与运行

```bash
cmake -S DCReg -B DCReg/build
cmake --build DCReg/build -j8

./DCReg/build/dcreg_minimal_example
./DCReg/build/dcreg_runner

cmake --build DCReg/build -j8 --target dcreg_parking_lot_example
./DCReg/build/dcreg_parking_lot_example
```

停车场 demo 需从 README Google Drive 下载 `prior_map.pcd` 至 `DCReg/data/Parking-Lot-example/`。

## 交叉链接

- [paper-dcreg-degenerate-lidar-registration 论文实体](../../wiki/entities/paper-dcreg-degenerate-lidar-registration.md)
- [GitHub 项目归档](../sites/dcreg-github.md)
- [论文归档](../papers/dcreg_ijrr_2026_hu.md)
