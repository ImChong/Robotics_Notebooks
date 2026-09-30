// 用 Chromium 把 deck.json 里的每一帧（幻灯片 + 分步显隐 + 字幕）截成 PNG。
// 用法：node render.cjs deck.json out_dir [slideIdFilter]
const { chromium } = require("playwright-core");
const fs = require("fs");
const path = require("path");

const CHROME = process.env.CHROME_PATH || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";

(async () => {
  const deck = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
  const outDir = process.argv[3];
  const only = process.argv[4] || "";
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ executablePath: CHROME, args: ["--no-sandbox", "--font-render-hinting=none"] });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  page.on("pageerror", e => console.error("pageerror:", e.message));
  await page.goto("file://" + path.resolve(__dirname, "template.html"));
  await page.evaluate(() => document.fonts.ready);
  const report = [];
  for (const s of deck.slides) {
    if (only && !s.id.startsWith(only)) continue;
    const todo = s.frames.filter(f => !fs.existsSync(path.join(outDir, f.file)) || process.env.FORCE);
    if (!todo.length) continue;
    const r = await page.evaluate(async (x) => window.loadSlide(x), s);
    if (r.overflow > 2 || r.sw > 2) report.push(`${s.id}: overflow v=${r.overflow} h=${r.sw}`);
    if (r.bad.length) report.push(`${s.id}: box overflow ${r.bad.join(", ")}`);
    // 首帧前多等一会儿，保证 webfont 子集全部就绪
    await page.waitForTimeout(120);
    await page.evaluate(() => document.fonts.ready);
    for (const f of todo) {
      await page.evaluate((y) => window.setFrame(y), f);
      await page.screenshot({ path: path.join(outDir, f.file), type: "png" });
    }
    process.stdout.write(".");
  }
  await browser.close();
  console.log("\n" + (report.length ? "OVERFLOW:\n" + report.join("\n") : "no overflow"));
})();
