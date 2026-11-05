---
type: entity
project_id: 3d-model-anything
project: https://wesbos.github.io/3d-model-anything/
code: https://github.com/wesbos/3d-model-anything
tags: [3d-reconstruction, lidar, threejs, real2sim, open-source]
status: complete
updated: 2026-10-29
summary: "3D Model Anything 结合 LiDAR、照片和编程智能体，迭代重建可编辑的 Three.js 室内场景。"
related:
  - ./paper-3dcodeverse.md
  - ./gitingest.md
sources:
  - ../../sources/repos/3d-model-anything.md
  - ../../sources/sites/3d-model-anything.md
---

# 3D Model Anything：室内场景重建

## 一句话定义

3D Model Anything 采集房间的 LiDAR 几何与照片，再由编程智能体借助测量和照片比对，迭代生成可编辑的 Three.js 场景。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LiDAR | Light Detection and Ranging | 采集空间几何 |
| AI | Artificial Intelligence | 编程智能体辅助场景建模 |
| API | Application Programming Interface | 场景和工具的程序接口 |

## 工作流程

1. 用带 LiDAR 的 iPhone 或 iPad 采集空间几何和照片并导出。
2. 使用仓库的 `scan-to-threejs` skill，让智能体根据测量构造场景。
3. 将相机对齐后的渲染与原图逐步比对并调整；浏览器支持多视角查看。

与 [3DCodeVerse](./paper-3dcodeverse.md) 侧重从提示生成并验证代码不同，此项目重点利用实景扫描证据重建房间。对 Real2Sim 资产准备有启发，但仓库介绍的是浏览器场景重建，不能推断输出已包含可用的机器人仿真碰撞体和物理参数。

## 工程实践与局限

- 代码、采集应用和 skill 已公开；代码使用 MIT 许可，示例素材另有条款。
- 这是需要智能体参与的迭代流程，不是一键生成精确模型；遮挡、测量和纹理误差需要人工检查。
- 官方示例不分发原始扫描和照片；示例页面中的扫描叠加功能因此不可用。
- 采集自己的场景需要支持 LiDAR 的设备；非 LiDAR 设备可导入与查看扫描。

## 源码运行时序图

入口：`npm run dev`（即 `vite --config apps/room-viewer/vite.config.ts`；`index.html` 加载 `apps/room-viewer/src/main.ts`，`play.html` 加载 `apps/room-viewer/src/play/main.ts`）

```mermaid
sequenceDiagram
  participant U as 浏览器用户
  participant M as apps/room-viewer/src/main.ts 查看器
  participant R as demos/index.ts ROOMS / roomById
  participant O4 as demos/office-v4/index.ts officeV4.build
  participant OL as demos/office-latest/index.ts officeLatest.build
  participant V2 as demos/office-v2/index.ts buildDetailedOffice
  participant RR as packages/room-runtime/render.ts 渲染管线
  participant P as packages/room-physics/physics.ts Physics

  U->>M: 打开 index.html（可用 ?room= 选择房间）
  M->>R: roomById(id)，默认 ROOMS[0]（office-v4）
  M->>M: 创建 WebGLRenderer、Scene、OrbitControls
  M->>RR: createPipeline(renderer, scene, camera)
  M->>O4: mod.build()（结果按房间 id 缓存为 Promise）
  O4->>OL: officeLatest.build()
  OL->>V2: buildDetailedOffice({ revision: 'october-7' })
  V2-->>OL: 房间 BuiltRoom
  OL-->>O4: room
  O4->>O4: updateDesk / updateRoom / updateArchitecture（office-v4 *-updates.ts）
  O4->>O4: 加载纹理，替换材质，调整灯光
  O4-->>M: BuiltRoom（group、lights、walls、camera）
  M->>M: scene.add(room.group, room.lights)
  M->>RR: prepareRendering：precompileRoomRendering → applyRoomRendering
  loop 有交互或 dirty 时 renderFrame
    M->>RR: 场景 MSAA 绘制 + GTAO 环境光遮蔽 + Bloom
    RR-->>U: 输出画面
  end

  Note over U,P: play.html 分支（src/play/main.ts）
  U->>M: 打开 play.html?room=office-v4
  M->>R: roomById(q.get('room'))
  M->>O4: roomModule.build()（与查看器共用同一房间构建）
  M->>P: new Physics(R, scene)，buildFromRoom(room, roomId)
  P-->>M: 刚体与凸包碰撞体就绪
  loop 每帧 setAnimationLoop
    M->>P: physics.update(dt)（仅在模拟时）
    M->>RR: 渲染当前帧
  end
```

图依据：wesbos/3d-model-anything@31b2dc9，实际读过 `package.json`、`apps/room-viewer/index.html`、`apps/room-viewer/vite.config.ts`、`apps/room-viewer/src/main.ts`、`apps/room-viewer/src/play/main.ts`（局部）、`demos/index.ts`、`demos/office-latest/index.ts`、`demos/office-v4/index.ts`、`packages/room-runtime/render.ts`、`packages/room-physics/physics.ts`（局部）。

## 关联页面

- [3DCodeVerse](./paper-3dcodeverse.md) — 可执行三维代码生成与验证。
- [Gitingest](./gitingest.md) — 整理仓库上下文以辅助阅读代码。

## 参考来源

- [官方仓库归档](../../sources/repos/3d-model-anything.md)
- [在线演示归档](../../sources/sites/3d-model-anything.md)
- 源码核对（2026-11-02）：wesbos/3d-model-anything@31b2dc9，读过 package.json、apps/room-viewer/index.html、apps/room-viewer/vite.config.ts、apps/room-viewer/src/main.ts、apps/room-viewer/src/play/main.ts（局部）、demos/index.ts、demos/office-latest/index.ts、demos/office-v4/index.ts、packages/room-runtime/render.ts、packages/room-physics/physics.ts（局部）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [官方仓库及使用说明](https://github.com/wesbos/3d-model-anything)
- [在线办公室演示](https://wesbos.github.io/3d-model-anything/)
