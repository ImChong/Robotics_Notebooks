# Panthera-HT Hub：官方产品与开发项目页

- **类型：** site / 官方项目主页
- **URL：** https://hightorque.cn/Panthera-HT_Hub/
- **机构：** 高擎动力（HighTorque Robotics）
- **入库日期：** 2026-11-18
- **代码：** https://github.com/HighTorque-Robotics/Panthera-HT_Main
- **扩展代码：** https://github.com/HighTorque-Robotics/Panthera-HT_lerobot
- **核对范围：** 已打开官网 Hub 并读取项目背景、产品亮点、通信、模型可视化、案例、LeRobot 数据采集—训练—部署链路与代码示例；对照主仓 README。
- **通读：** 已完成
- **对应项目：** [Panthera-HT](../../wiki/entities/cn-os-panthera-ht-main.md)

## 官网页面可核实内容

- 官网页面说明项目源于 wEch1ng 和伙伴参与的开源 Panthera 机械臂，后由高擎合作完善为 Panthera-HT。
- 产品亮点标示 6 轴、860 mm 臂展、0.1 mm 重复定位精度、CAN-FD 通信，支持 ROS 2、LeRobot、C++ / Python 和 URDF；**精度和臂展为厂商标示，非独立实测**。
- LeRobot 展示为“采集 → 策略训练 → 真机部署”，并示意 PantheraLeader.get_action() → PantheraFollower.send_action()。
- **显示限制：** 官网“技术参数”首屏爬取文本为 0dof、0kg、0mm、0Nm 等动态加载初值，不能把它们当产品实际参数；本页只保留官网其他明确文字说明中的数字。
- 能力演示与模型浏览器不是第三方复现验证；不将演示能力误写成已量化的基准结果。

## 开源核查

官方网站指向官方 GitHub 适配代码；已公开的机械 CAD / SDK / ROS 2 / LeRobot 等具体资产以各自仓库和版本为准。项目主页的“24 小时跑通策略”为宣传说明，不代表通用的部署工期保证。
