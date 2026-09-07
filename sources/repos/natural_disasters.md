# natural-disasters / ABYSSAL（浏览器程序化海洋与极端天气）

> 来源归档

- **标题：** natural-disasters（项目名 ABYSSAL）
- **类型：** repo
- **链接：** https://github.com/Token-Gremlin/natural-disasters
- **演示：** https://token-gremlin.github.io/natural-disasters/
- **机构：** Token-Gremlin（个人/独立开源）
- **许可：** MIT
- **Stars：** ~232（2026-09-07）
- **技术栈：** Three.js r169、WebGL2、GLSL3、Vite；零外部纹理/网格资产
- **入库日期：** 2026-09-07
- **一句话说明：** 浏览器内全 GPU 程序化海洋（多级联 FFT JONSWAP 海浪）+ 体积云/大气散射 + 飓风/海啸/水龙卷等灾害场；约 400 KB JS/GLSL，可作仿真可视化与程序化环境技术参考。
- **沉淀到 wiki：** [natural-disasters-abyssal](../../wiki/entities/natural-disasters-abyssal.md)

---

## 开源状态（2026-09-07）

| 组件 | 状态 |
|------|------|
| 源码 | **已开源**（MIT） |
| 在线 Demo | GitHub Pages 自动部署 `main` |
| 运行时依赖 | 构建后静态 bundle，无服务端 |

---

## 核心摘录

### 1) 海洋：多级联 FFT + 物理着色

- JONSWAP 谱 + 方向扩散；GPU butterfly IFFT 每帧演化 swell / 风浪 / 涟漪。
- 泡沫由表面 Jacobian 与波陡累积；PBR 水体（GGX、Fresnel、SSS、各向异性粗糙度）。
- 屏幕空间投影网格 + 射线与解析灾害场求交，避免海啸墙撕裂。

**对 wiki 的映射：** [natural-disasters-abyssal](../../wiki/entities/natural-disasters-abyssal.md)、[procedural-terrain-generation](../../wiki/concepts/procedural-terrain-generation.md)

### 2) 天空与天气

- Bruneton/Hillaire 风格大气散射 LUT；Perlin-Worley 体积云 + 天气图驱动细胞簇。
- 灾害：暴雨、喷雾、体积闪电、水龙卷、飓风眼墙、 Rogue wave、不对称海啸浅水变形；均 **变形真实水面**。

**对 wiki 的映射：** [natural-disasters-abyssal](../../wiki/entities/natural-disasters-abyssal.md)、[domain-randomization](../../wiki/concepts/domain-randomization.md)（极端天气 DR 叙事参照）

### 3) 工程入口

```bash
git clone https://github.com/Token-Gremlin/natural-disasters.git
cd natural-disasters && npm install && npm run dev
# Node 20.19+ 或 22.12+；npm run build → dist/ 静态托管
```

两种模式：**cinematic** 自动风暴序列；**sandbox** 自由飞行并触发灾害事件。

**对 wiki 的映射：** [natural-disasters-abyssal](../../wiki/entities/natural-disasters-abyssal.md)

---

## 与机器人研究的邻接读法

- **不是** 物理引擎或 RL 环境——无刚体/接触动力学；价值在 **程序化环境场** 与 **GPU 实时渲染管线**（FFT 海面、体积天气）可作为 sim 可视化、海事/两栖机器人 **外观级** 场景生成参考。
- 与 [InfiniteDiffusion Terrain Diffusion](../papers/infinite_diffusion_terrain.md) 等 **程序化户外几何** 同属「无手工资产的大尺度场」方向，但本仓侧重 **流体/大气实时着色** 而非 RL 训练接口。

---

## 对 wiki 的映射

- **wiki/entities/natural-disasters-abyssal.md** — 独立实体页
- **wiki/concepts/procedural-terrain-generation.md** — 程序化环境交叉引用（可选后续补链）

## 当前提炼状态

- [x] README 与 demo 核查
- [x] 开源状态（MIT + Pages）
- [x] wiki 实体页
