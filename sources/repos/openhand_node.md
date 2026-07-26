# openhand_node

> 来源归档

- **标题：** openhand_node — Yale OpenHand ROS / Python 控制
- **类型：** repo
- **来源：** Yale Grab Lab（`grablab`）
- **链接：** <https://github.com/grablab/openhand_node>
- **项目页：** <https://www.eng.yale.edu/grablab/openhand/>
- **CAD 配套：** <https://github.com/grablab/openhand-hardware>
- **License：** MIT
- **Stars：** ~13（2026-07-26 快照）
- **最近推送：** 2022-02-10
- **入库日期：** 2026-07-26
- **一句话说明：** OpenHand **活跃控制仓**：Python 对象 + ROS 节点，支持 **Model O / T / T42**；兼容 Dynamixel Protocol 1（RX/MX）与 Protocol 2（X 系列）；经 U2D2 + 外供电 hub 通信。
- **沉淀到 wiki：** 是 → [`wiki/entities/yale-openhand.md`](../../wiki/entities/yale-openhand.md)（工程实践 / 控制栈）

---

## 核心能力（README）

- 替换已弃用的 [`openhand-software`](https://github.com/grablab/openhand-software)。
- 纯 Python 控制入口：`src/openhand_node/hands.py`、`lib_robotis_mod.py`、`registerDict.py`。
- ROS（文档称已在 Kinetic 测过）节点便于集成。
- 依赖：`pyserial`、`numpy`、`scipy`；电机需 **约 12 V** 外供（U2D2 **不能**给电机供电）。

## 支持范围边界

| 已支持（README） | 说明 |
|------------------|------|
| Model O / T / T42 | 主路径；后续型号「将支持」需自行联系或扩展 |
| **Model F3** | **未**在 README 列为已支持；硬件为 T42 改编，控制可参考 T42 接口但需自行验证 ID/行程 |

## 对 wiki 的映射

- [Yale OpenHand](../../wiki/entities/yale-openhand.md)
- CAD：[`openhand-hardware.md`](openhand-hardware.md)
