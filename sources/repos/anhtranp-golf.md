# Kinematic Vision AI

> 来源归档

- **标题：** Kinematic Vision AI — Premium AI Golf Motion Dashboard
- **类型：** repo
- **代码：** <https://github.com/anhtranp/golf>
- **项目页：** README 与界面截图位于仓库；未发现单独部署的项目站点
- **来源线索：** 用户提供的 2026-10-16 截图（GitHubStore 视频画面）
- **入库日期：** 2026-10-16
- **一句话说明：** 浏览器端高尔夫挥杆演示工具，用 MediaPipe 在本机摄像头画面上做姿态跟踪，提取挥杆运动学指标，并可把指标摘要交给 Gemini 2.5 Flash 生成分阶段建议。
- **开源状态：** **代码公开**。仓库可访问；未找到独立项目站点、论文、数据集或模型权重。README 声称 MIT，但 GitHub 未识别许可证且仓库根目录没有 LICENSE 文件，使用许可需向作者确认。

---

## 功能与实现摘要

- 以 React、TypeScript、Vite 构建浏览器应用；通过 `@mediapipe/tasks-vision` 在浏览器内执行 PoseLandmarker，并绘制骨架与角度遥测。
- 将挥杆划分为 Address、Backswing、Top、Downswing、Impact、Follow-through 六阶段，展示肩部转角、髋部旋转、躯干倾角等指标。
- 配置 Gemini API Key 后，把采集到的运动学数值与阶段摘要发送给 Gemini 2.5 Flash，返回阶段评分、PGA 基准差值和训练建议；无 Key 或请求失败时使用本地规则分析。
- README 声称视频仅在浏览器内处理。当前 AI 服务代码发送的是数值与阶段摘要，不发送视频帧；使用 Gemini 时这些衍生数据会传至 Google API。
- 支持全屏练习界面、倒计时采集，并在摄像头不可用时切换为演示运动学数据。

## 运行入口

- **依赖：** Node.js、npm 或 pnpm；README 写明 Node.js 18+
- **开发：** `npm install` 后运行 `npm run dev`，打开本地 Vite 地址（README 示例为 `http://localhost:8443/`，端口以终端实际输出为准）。
- **构建：** `npm run build`
- **代码入口：** `src/components/CameraMotionTracker.tsx`、`src/utils/poseTracker.ts`、`src/utils/aiCoachService.ts`

## 复现边界与注意事项

- README / PRD 中的“60 FPS”“PGA Tour benchmark”等属于项目说明；仓库没有公开验证数据或基准来源，不能据此视为经运动科学验证的测量系统。
- 当前代码读取的是 MediaPipe 人体姿态关键点；README 中的杆速等指标含估算或演示回退值，不等同于测距仪或高速摄影的实测。
- 仓库 README 的 clone 示例仍是占位地址 `your-username/premium-ai-golf-dashboard.git`，实际仓库地址以本条代码链接为准。
- 项目 README 标注 MIT，但仓库没有许可证文件；在许可补齐前，不应把该声明当作已核实的授权。

## 对 wiki 的映射

- 作为人类动作姿态跟踪与浏览器端推理的应用示例；当前没有独立研究结果或足够的机器人领域专属方法，暂留在原始来源层。
