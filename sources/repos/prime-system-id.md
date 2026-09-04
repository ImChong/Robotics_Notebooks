# PRIME

> 来源归档（仓库 + RSS 2026 论文）

- **标题：** PRIME: Physically-consistent Robotic Inertial and Motion Estimation
- **类型：** repo
- **技术路线分组：** 系统辨识 / 接触一致运动重建 / 工程与实机部署
- **链接：** https://github.com/well-robotics/PRIME
- **项目页：** https://jkangkjr.github.io/PRIME-project/
- **arXiv：** https://arxiv.org/abs/2605.17681
- **许可：** BSD-3-Clause（保留上游 Crocoddyl 归属）
- **入库日期：** 2026-07-30（HMI 开源主表）
- **复核日期：** 2026-09-04（RSS 论文 + 项目页 + README 实验入口）
- **一句话说明：** 从关节状态、速度和执行器命令反推动力学一致轨迹、接触力与惯性参数；C++ 实现复用 Crocoddyl/FDDP。
- **开源状态（2026-09-04）：** **已开源** — 仓内 `experiments/G1_real_dance_*`、`Go2_*` 可 CMake 编译运行；非占位 README。
- **策展入口：** [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)
- **沉淀到 wiki：** 是 → [`wiki/entities/prime-system-id.md`](../../wiki/entities/prime-system-id.md)

## 为什么值得保留

主表将该项列为人形运动智能工程入口；RSS 2026 论文把同一仓库升级为 **接触隐式 MAP 惯量辨识** 的可复现实现。日志看似正常却无法解释接触或载荷时，可用于数据清洗与系统辨识。

## 运行入口（README）

```bash
cmake -S . -B build -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_EXAMPLES=OFF
cmake --build build --target g1_real_dance_1 -j2
build/experiments/G1_real_dance_1/g1_real_dance_1 \
  experiments/G1_real_dance_1/config/g1_real_dance_1.xml
```

输出含 `inertia_identification.txt`、`xs_results_fddp.csv`、`f_rollout.csv`。

## 对 wiki 的映射

- [PRIME](../../wiki/entities/prime-system-id.md)
- [人形整机闭环惯量标定](../../wiki/concepts/humanoid-closed-loop-inertia-calibration.md)
- [Crocoddyl](../../wiki/entities/crocoddyl.md)
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
