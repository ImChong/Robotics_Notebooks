# Eclipse Cyclone DDS 官网（cyclonedds.io）

> 来源归档

- **标题：** Eclipse Cyclone DDS — Home / Documentation
- **类型：** site（官方项目页 + 文档入口）
- **来源：** Eclipse Foundation / Eclipse IoT（`iot.cyclonedds`）
- **链接：** https://cyclonedds.io/
- **文档索引：** https://cyclonedds.io/docs/
- **代码：** https://github.com/eclipse-cyclonedds/cyclonedds（已开源，见 [repos/cyclonedds.md](../repos/cyclonedds.md)）
- **Eclipse 项目页：** https://projects.eclipse.org/projects/iot.cyclonedds
- **入库日期：** 2026-07-28
- **一句话说明：** 高性能、可互操作的 OMG DDS 实现项目入口；强调低延迟/低抖动、发现开销聚合、DDS-Security 与 ROS 2 tier-1 中间件定位。
- **沉淀到 wiki：** 是 → [`wiki/entities/cyclone-dds.md`](../../wiki/entities/cyclone-dds.md)、[`wiki/concepts/dds-communication.md`](../../wiki/concepts/dds-communication.md)

## 为什么值得保留

- Unitree SDK2 / `unitree_ros2` 等真机栈默认走 **CycloneDDS**；选型与域/QoS 调试需要官方定位而非二手摘要。
- 官网明确 **DDSI-RTPS + DDS-Security** 互操作、可与其他合规 DDS / Zenoh 等桥接。
- `cyclonedds.io/docs/` 按版本（11.0 / 0.10 / master 等）索引 Main / C++ / Python / Insight 文档——与 Unitree「钉定 0.10.x」实践直接相关。

## 开源核查（2026-07-28）

| 项 | 状态 |
|----|------|
| 官网 → GitHub | ✅ [eclipse-cyclonedds/cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds) |
| 代码开放度 | **已开源**（EPL-2.0 **或** EDL-1.0） |
| 绑定仓 | `cyclonedds-cxx`、`cyclonedds-python` 等同组织 sibling repos |

## 官网自述要点（摘录）

1. **Fast & Dependable**：低延迟、高吞吐；嘈杂网络下抖动小； footprint 适合嵌入式与企业侧。  
2. **Consistent & Scalable**：聚合发现代表以减发现开销；可按组播/单播分流；拓扑变化下保持一致性。  
3. **Secure & Interoperable**：预置或自定义认证/授权/加密插件（RSA、DH、AES-GCM/GMAC 等）；与其他合规 DDS 互通。

## 文档版本入口（2026-07）

| 系列 | 说明 |
|------|------|
| Latest (master) | 滚动主线文档 |
| **11.0**（*Marche des Davidsbündler…*） | 当前发行线（例 11.0.1） |
| **0.10**（*Lettres Dansantes*） | 老发行线；Unitree Foxy 等常钉此系 |
| 0.9 / 0.8 | 更早发行线 |

完整 URL 以 https://cyclonedds.io/docs/ 为准。

## 对 wiki 的映射

- 实体：[cyclone-dds](../../wiki/entities/cyclone-dds.md)
- 概念：[dds-communication](../../wiki/concepts/dds-communication.md)
- 厂商实践：[unitree-ros2](../../wiki/entities/unitree-ros2.md)、[unitree-sdk2](../../wiki/entities/unitree-sdk2.md)
- 规范：[omg-dds-spec](omg-dds-spec.md)
- 代码仓：[repos/cyclonedds.md](../repos/cyclonedds.md)
