# OpenVINO 官网与文档门户

> 来源归档（以 www.openvino.ai 与 docs.openvino.ai 2026.3 叙述为准）

- **标题：** Intel® Distribution of OpenVINO™ Toolkit
- **类型：** site / docs / deployment
- **官网：** https://www.openvino.ai/
- **文档：** https://docs.openvino.ai/
- **博客：** https://blog.openvino.ai/
- **代码：** https://github.com/openvinotoolkit/openvino
- **入库日期：** 2026-09-09
- **一句话说明：** OpenVINO 的 **产品门户 + 用户文档** 入口：官网强调「write once, deploy anywhere」与 GenAI/LLM 可及性；文档站（**2026.3**）按 **Base / GenAI / Physical AI / Model Server** 四条产品线组织，并提供安装、框架兼容、NNCF 压缩、性能 benchmark 与 Physical AI 机器人部署教程。
- **沉淀到 wiki：** [OpenVINO](../../wiki/entities/openvino.md)

---

## 官网要点（www.openvino.ai）

1. **定位**：开源 AI 工具包，在保持精度的前提下 **降低延迟、提高吞吐、缩小模型 footprint、优化硬件利用**。
2. **覆盖域**：计算机视觉、LLM、生成式 AI；支持 TensorFlow / PyTorch 等框架模型转换与跨 **Intel 硬件 + 云/边/端/浏览器** 部署。
3. **2026.3 更新**：首页突出 What's New；提供 AI Programming Workshops（GenAI、LLM、AI PC 等 Jupyter 工作坊）。
4. **AI Reference Kits**：端到端开源样例应用 + 可定制 Jupyter Notebook，面向快速部署。
5. **社区与支持**：GitHub Issues、Intel DevHub Discord、Stack Overflow `openvino` 标签。

---

## 文档门户要点（docs.openvino.ai 2026.3）

### 四大工具产品线

| 产品 | 用途 |
|------|------|
| **OpenVINO Base Package** | 常规 AI 模型推理 |
| **OpenVINO GenAI** | 生成式模型部署 |
| **OpenVINO Physical AI** | **机器人 VLA 等 Physical AI 部署**（专章 `/2026/physical-ai.html`） |
| **OpenVINO Model Server** | 服务端推理（OVMS） |

### 入门路径

- **安装**：多平台分发（pip / conda / 系统包）；GenAI 与 Physical AI 有独立安装页
- **性能 Benchmark**：官方模型与硬件对照表
- **框架兼容**：TensorFlow、ONNX、PaddlePaddle 直连或转 OpenVINO 格式
- **轻松部署**：数行代码即可推理
- **服务化**：OVMS 面向微服务 / Kubernetes
- **模型压缩**：NNCF 后训练与训练时压缩

### 关键特性（文档归纳）

- **一次编写、多设备部署**：自动设备发现；Linux / Windows / macOS；Python / C++ / C API
- **轻量部署**：最小外部依赖；可按模型定制编译减小二进制
- **快速启动**：**CPU 先行编译 + 异步切换 GPU/NPU** 降低首帧延迟；模型缓存加速冷启动
- **PyTorch 2.0**：`torch.compile` OpenVINO 后端
- **Hugging Face**：预优化 OpenVINO 模型集，免手动转换

### Physical AI 专章（2026）

文档路径：`/2026/physical-ai.html`

**工作流：**

```
exported policy package → InferenceModel → PolicyRuntime → Robot
```

**Python 最小示例：**

```python
from physicalai.inference import InferenceModel
from physicalai.runtime import PolicyRuntime, SyncExecution
from physicalai.robot import SO101
from physicalai.capture import UVCCamera

model = InferenceModel("./exports/act_policy")
robot = SO101(port="/dev/ttyACM0")
cameras = {"wrist": UVCCamera(device="/dev/video0", width=640, height=480)}

runtime = PolicyRuntime(
    fps=30, robot=robot, model=model, cameras=cameras,
    execution=SyncExecution(),
)
with runtime:
    runtime.run(duration_s=60)
```

**机器人协议**（结构类型，无需继承）：

- `connect()` / `disconnect()` / `get_observation()` / `send_action()`
- `joint_names` 与 action 向量顺序对齐
- observation 至少含 `joint_positions`、`timestamp`；可选 `sensor_data`、`images`

**Intel 产品页补充**（physical-ai.html on intel.com）：

- 统一 VLA 接口：ACT、SmolVLA、PI0.5 等通过单参数切换架构
- **Physical AI Studio** 集成：Intel 优化预验证 VLA 模型
- **LeRobot 集成**：导出 LeRobot 模型，PyTorch 或 OpenVINO 推理

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [OpenVINO 主仓](../repos/openvino.md) | 运行时与转换器源码 |
| [Physical AI 运行时仓](../repos/openvino-physicalai.md) | 机器人部署侧 Python 包 |
| [ONNX Runtime](../../wiki/entities/onnxruntime.md) | ORT OpenVINO EP 文档互指 |
| [VLA](../../wiki/methods/vla.md) | Physical AI 专章直接服务 VLA onboard |

---

## 对 wiki 的映射

- 更新 **`wiki/entities/openvino.md`**：补齐 2026.3 产品线、Physical AI 工作流与工程实践表
- 交叉 **`sources/repos/openvino-official.md`**（文档索引续更）

---

## 外部参考

- [OpenVINO 官网](https://www.openvino.ai/)
- [OpenVINO 文档](https://docs.openvino.ai/)
- [Physical AI 文档](https://docs.openvino.ai/2026/physical-ai.html)
- [Intel Physical AI 产品页](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/physical-ai.html)
- [OpenVINO Blog](https://blog.openvino.ai/)
