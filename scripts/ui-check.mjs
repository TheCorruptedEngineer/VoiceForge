// Headless UI smoke check (house pattern: Brave + puppeteer-core).
// Reuses puppeteer-core from the Corrupted Industries node_modules.
// Run: node scripts/ui-check.mjs
import { createRequire } from "module";
const require = createRequire(
  "C:/Users/AdministratorCorrupt/Desktop/Claude Projects/Corrupted Industries/package.json");
const puppeteer = require("puppeteer-core");

const BRAVE = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe";
const URL = "http://127.0.0.1:8484/";

const browser = await puppeteer.launch({ executablePath: BRAVE, headless: "new" });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", e => errors.push("pageerror: " + e.message));
page.on("console", m => { if (m.type() === "error") errors.push("console: " + m.text()); });

await page.goto(URL, { waitUntil: "networkidle2", timeout: 20000 });
await new Promise(r => setTimeout(r, 1500));

const checks = await page.evaluate(() => ({
  title: document.title,
  engineOptions: document.querySelectorAll("#engine-sel option").length,
  modelOptions: document.querySelectorAll("#model-sel option").length,
  paramSliders: document.querySelectorAll("#params .param").length,
  voiceCards: document.querySelectorAll(".voice-card").length,
  historyItems: document.querySelectorAll(".hist").length,
  generateBtn: !!document.getElementById("btn-generate"),
}));

console.log(JSON.stringify(checks, null, 2));
let fail = errors.length > 0 || checks.engineOptions < 2 || checks.paramSliders < 1
  || !checks.generateBtn || checks.historyItems < 1;
if (errors.length) console.log("JS errors:", errors);
await browser.close();
console.log(fail ? "UI CHECK FAILED" : "UI CHECK PASSED");
process.exit(fail ? 1 : 0);
