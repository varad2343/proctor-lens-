// End to end (master spec 15.1): a proctor creates a session, a candidate takes it with a fake camera, events reach
// the dashboard. Real models, real browser; nothing mocked. Needs a running `proctorlens serve`, Edge (or set
// CHANNEL=chrome), and a fake camera from `python tools/fake_cam.py fake_cam.y4m`.
//   PROCTORLENS_PASSWORD=... FAKE_CAM=/abs/fake_cam.y4m [SHOTS=dir] [EXAM_S=45] [BASE=http://127.0.0.1:8000] node e2e.mjs
import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://127.0.0.1:8000";
const EXAM_S = Number(process.env.EXAM_S ?? 45);
const shot = (page, name) => process.env.SHOTS && page.screenshot({ path: `${process.env.SHOTS}/${name}.png`, fullPage: true });
const log = (s) => console.log(`- ${s}`);
const text = async (loc) => (await loc.innerText()).replace(/\s+/g, " ").trim();

const browser = await chromium.launch({
  channel: process.env.CHANNEL ?? "msedge",
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    ...(process.env.FAKE_CAM ? [`--use-file-for-fake-video-capture=${process.env.FAKE_CAM}`] : [])],
});
try {
  const proctor = await (await browser.newContext({ viewport: { width: 1360, height: 900 } })).newPage();
  await proctor.goto(`${BASE}/#/login`);
  await proctor.getByLabel("Username").fill(process.env.PROCTORLENS_USER ?? "proctor");
  await proctor.getByLabel("Password").fill(process.env.PROCTORLENS_PASSWORD ?? "");
  await proctor.getByRole("button", { name: "Log in" }).click();
  await proctor.getByRole("heading", { name: "Sessions" }).waitFor();
  await proctor.getByLabel("Candidate (pseudonym)").fill("E2E");
  await proctor.getByRole("button", { name: "New session" }).click();
  const join = (await proctor.locator("code").first().textContent()).trim();
  const sid = join.split("/").at(-2);
  log(`proctor logged in, session ${sid} created`);
  await shot(proctor, "1-sessions");

  const cand = await (await browser.newContext({ viewport: { width: 1280, height: 800 } })).newPage();
  await cand.goto(join);
  await shot(cand, "2-consent");
  await cand.getByRole("checkbox").check();
  await cand.getByRole("button", { name: "Continue" }).click();
  await cand.getByText("Picture quality").waitFor({ timeout: 30000 });
  await cand.waitForTimeout(1500);
  log(`camera check: ${await text(cand.locator("section").first())}`);
  await shot(cand, "3-camera-check");
  await cand.getByRole("button", { name: "Continue to calibration" }).click();
  await cand.getByRole("button", { name: "Start" }).click();
  await cand.waitForTimeout(3000);
  await shot(cand, "4-calibration-dot");
  await cand.getByRole("heading", { name: /^Calibration (complete|finished)$/ }).waitFor({ timeout: 90000 });
  log(`calibration: ${await text(cand.locator("section").first())}`);
  await shot(cand, "5-calibration-result");
  await cand.getByRole("button", { name: "Continue", exact: true }).click();
  await cand.getByRole("heading", { name: "Mock exam" }).waitFor({ timeout: 60000 });
  await cand.getByLabel("42", { exact: true }).check();
  await cand.evaluate(() => window.dispatchEvent(new Event("blur"))); // focus lost for 1.5 s, then a paste
  await cand.waitForTimeout(1500);
  await cand.evaluate(() => (window.dispatchEvent(new Event("focus")), document.dispatchEvent(new Event("paste"))));
  log(`exam running for ${EXAM_S} s (fake camera: face / covered / empty room / several people, looping)`);
  await shot(cand, "6-exam");
  await proctor.goto(`${BASE}/#/live/${sid}`);
  await cand.waitForTimeout((EXAM_S * 1000) / 2);
  await shot(proctor, "7-live");
  await cand.waitForTimeout((EXAM_S * 1000) / 2);
  await cand.getByRole("button", { name: "Submit", exact: true }).click();
  await cand.getByRole("button", { name: "Confirm: submit answers" }).click();
  await cand.getByRole("heading", { name: "Submitted" }).waitFor({ timeout: 60000 });
  await shot(cand, "8-submitted");

  const events = await (await proctor.request.get(`${BASE}/api/sessions/${sid}/events`)).json();
  const count = {};
  for (const e of events) count[e.type] = (count[e.type] ?? 0) + 1;
  log(`events: ${JSON.stringify(count)}; with clip ${events.filter((e) => e.clip_path).length}/${events.length}`);
  await proctor.goto(`${BASE}/#/review/${sid}`);
  await proctor.getByText("Export report").waitFor();
  await proctor.waitForTimeout(2000);
  await shot(proctor, "9-review");
  const report = await proctor.request.get(`${BASE}/api/sessions/${sid}/report`);
  log(`report: HTTP ${report.status()}, ${(await report.text()).length} bytes`);
  if (!events.some((e) => e.type !== "BROWSER_INTEGRITY")) throw new Error("no camera events reached the dashboard");
  if (!count.BROWSER_INTEGRITY) throw new Error("browser telemetry did not arrive");
  log("PASS");
} finally {
  await browser.close();
}
