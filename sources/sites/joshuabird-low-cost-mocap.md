# Open Source Motion Capture for Autonomous Drones

> 来源归档（ingest）

- **标题：** Open Source Motion Capture for Autonomous Drones
- **作者：** Joshua Bird
- **类型：** 项目博客
- **链接：** <https://joshuabird.com/blog/post/mocap-drones>
- **发布日期：** 2023-11-20
- **代码：** <https://github.com/jyjblrd/Low-Cost-Mocap>
- **开源状态：** 已开源。官方仓库包含相机端处理代码、ESP32 收发固件和相机支架 3D 文件；仓库为 MIT 许可证。
- **一句话说明：** 用多台改装 PS3 Eye 红外相机进行室内 outside-in 多目标跟踪，并把位置数据发送给微型无人机，实现自主飞行演示。

## 核心摘录

- **定位管线：** 多台外部相机观测无人机上的红外 LED，以多视角几何恢复房间坐标中的目标位置；相机位姿可通过 Structure from Motion 与 bundle adjustment 自动估计。
- **硬件成本：** 作者报告相机动捕系统约 20 美元；这是相机系统估算，不包括无人机。演示用无人机本身约 50 美元。
- **相机改装：** PS3 Eye 移除红外截止滤镜并加装红外滤光片，以观察机载红外标记。作者指出原镜头改装挑相机型号且操作较繁琐，也可替换无红外截止滤镜的新镜头。
- **闭环演示：** 主机运行视觉定位后将状态传至 ESP32；无人机侧以 ESP32 模拟 RC 接收端，配合飞控实现室内定位悬停和飞行。
- **可复现入口：** README 要求 Python 的 pseyepy、带 SFM 模块的 OpenCV，以及 Node/Yarn；前端运行 `yarn run dev`，后端运行 `python3 api/index.py`。
- **局限：** 作者说明仓库文档仍不完善；其毫米级定位描述是项目报告的 room-scale 精度，部署时仍需独立验证标定、遮挡、时延和飞行稳定性。

## 对 wiki 的映射

- [Low Cost Mocap 实体页](../../wiki/entities/low-cost-mocap.md)
- [代码仓库归档](../repos/low-cost-mocap.md)
