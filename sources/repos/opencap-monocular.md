# OpenCap Monocular

> 来源归档

- **标题：** OpenCap Monocular
- **类型：** repo / 生物力学软件
- **机构：** University of Utah · Movement Bioengineering Lab (MoBL)
- **链接：** https://github.com/utahmobl/opencap-monocular
- **项目页：** https://utahmobl.github.io/OpenCap-monocular-project-page/
- **论文：** https://arxiv.org/abs/2603.24733
- **产品：** https://opencap.ai
- **入库日期：** 2026-06-30
- **许可证：** PolyForm Noncommercial License 1.0.0（商业使用需另行协议）
- **一句话说明：** 单目静态手机视频 → WHAM 初始化 → 物理约束优化 → OpenSim IK → 运动学/动力学导出；与双相机 OpenCap 同属临床可扩展生物力学评估栈。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-opencap-monocular.md`](../../wiki/entities/paper-opencap-monocular.md)

---

## 管线（README）

1. 视频预处理与旋转校正
2. **WHAM** 3D 姿态估计
3. 相机外参与姿态 **优化精炼**
4. **OpenSim** 逆运动学与导出
5. 可视化（`mono.json` → [OpenCap Visualizer](https://visualizer.opencap.ai)）

## 输出产物

| 文件 | 用途 |
|------|------|
| `mono.json` | OpenCap 在线可视化 |
| `*.trc`, `*.mot` | OpenSim 标准格式 |
| `*_scaled.osim` | 缩放后的 OpenSim 骨骼模型 |

## 安装要点

- Ubuntu 20.04/22.04、Python 3.9、NVIDIA driver ≥520
- `git clone --recursive`；详见 `installation/INSTALL_SLIM.md`

## 关联仓库（动力学后处理）

| 仓库 | 说明 |
|------|------|
| [opencap-org/opencap-processing](https://github.com/opencap-org/opencap-processing) | 运动学后处理 + 肌肉骨骼动力学仿真 |
| [opencap-org/opencap-processing-grf](https://github.com/opencap-org/opencap-processing-grf) | 行走地面反力混合 ML–仿真 |

## 上游依赖与许可注意

| 组件 | 许可 | 备注 |
|------|------|------|
| WHAM | MIT | 3D 姿态初始化 |
| ViTPose | Apache 2.0 | 2D 关键点 |
| SMPL | MPI 自定义 | **非商业科研**；需单独注册下载 |
| 本仓库 | PolyForm Noncommercial | 商业需联系作者 |

## 对 wiki 的映射

- 论文摘录：[opencap_monocular_arxiv_2603_24733.md](../papers/opencap_monocular_arxiv_2603_24733.md)
- 项目页：[opencap-monocular-github-io.md](../sites/opencap-monocular-github-io.md)
- 实体页：[wiki/entities/paper-opencap-monocular.md](../../wiki/entities/paper-opencap-monocular.md)
