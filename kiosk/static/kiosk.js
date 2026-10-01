/* E.P.I.V.U.E. kiosk page. One state machine, one fetch per scan.
   States: idle -> live -> captured -> reading -> result | refused | error
   The camera is either the Pi's ("pi": the server streams /preview.mjpg and
   takes the photo), the browser's ("webcam": getUserMedia on a laptop), or
   absent ("off": pick a file). The server decides nothing about the UI; it
   answers /scan with a verdict and this file puts the words on screen. */

const $ = (id) => document.getElementById(id);
const body = document.body;
const S = { camera: "off", photoUrl: null, outcome: null, stream: null, quitArmed: 0 };

// ---------- boot ----------
async function boot() {
  try {
    const h = await (await fetch("/health")).json();
    S.camera = h.camera ? "pi" : (navigator.mediaDevices && navigator.mediaDevices.getUserMedia ? "webcam" : "off");
    $("model-line").textContent = h.ok ? `${h.model.file} · ${h.model.tta ? "4 views" : "1 view"}` : "model not loaded";
  } catch (e) {
    S.camera = "off";
  }
  body.dataset.camera = S.camera;
  render("idle");
}

// ---------- rendering ----------
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}
function button(label, onClick, primary) {
  const b = el("button", "btn" + (primary ? " primary" : ""), label);
  b.type = "button";
  b.addEventListener("click", onClick);
  return b;
}
function rows(items, numbered) {
  const box = el("div", "mid");
  items.forEach((t, i) => {
    const r = el("div", "row");
    if (numbered) r.appendChild(el("span", "row-num", String(i + 1)));
    r.appendChild(el("span", "", t));
    box.appendChild(r);
  });
  return box;
}
function setPage({ eyebrow, headline, lede, mid, actions, tone, mark }) {
  body.dataset.tone = tone || "";
  $("eyebrow").textContent = eyebrow || "";
  $("headline").textContent = headline || "";
  $("lede").textContent = lede || "";
  const m = $("mid");
  m.replaceChildren();
  if (mark) m.appendChild(el("div", "mark"));
  if (mid) m.appendChild(mid);
  $("actions").replaceChildren(...(actions || []));
}

function render(state, data) {
  body.dataset.state = state;
  const cam = S.camera;
  if (state !== "live") stopWebcam();

  if (state === "idle") {
    forgetPhoto();
    $("hint").textContent = cam === "off" ? "No camera on this machine — choose a picture to try it." : "Camera ready";
    $("readout").textContent = "";
    setPage({
      eyebrow: "Skin check · takes a few seconds",
      headline: "Check a spot on your skin",
      lede: "Nothing you photograph leaves this device. No name, no account.",
      mid: rows(["Fill the ring with the spot and hold still", "Tap once to take the photo", "Read what to do next, in plain words"], true),
      actions: cam === "off"
        ? [button("Choose a picture", pickFile, true)]
        : [button("Start a skin check", startLive, true), button("Choose a picture", pickFile)],
    });
  }

  if (state === "live") {
    $("readout").textContent = cam === "pi" ? "LIVE · Pi camera" : "LIVE · this computer's camera";
    setPage({
      eyebrow: "Step 1 of 2 · frame the spot",
      headline: "Fill the ring, then hold still",
      lede: "Rest the camera on the skin so the spot sits in the middle of the ring.",
      mid: rows(["Spot inside the ring, edges included", "Even light — no shadow, no glare", "Two seconds still, then tap"]),
      actions: [button("Take the photo", takePhoto, true), button("Choose a picture", pickFile)],
    });
  }

  if (state === "captured") {
    $("readout").textContent = "Photo taken";
    setPage({
      eyebrow: "Step 2 of 2 · check the photo",
      headline: "Is the spot inside the ring?",
      lede: "If it is not, take another. If it is, start the check.",
      actions: [button("Check this spot", () => scan(false), true), button("Take another", retake)],
    });
  }

  if (state === "reading") {
    $("readout").textContent = "Reading…";
    const steps = el("div", "steps");
    ["Checking the photo and finding skin", "Looking for one spot with a clear edge", "Comparing it with the on-device model"].forEach((t) => {
      const s = el("div", "step");
      s.appendChild(el("i"));
      s.appendChild(el("span", "", t));
      steps.appendChild(s);
    });
    setPage({
      eyebrow: "Reading",
      headline: "Reading the spot…",
      lede: "Everything happens on this device. No internet needed.",
      mid: steps,
      actions: [],
    });
    tickSteps(steps);
  }

  if (state === "result") {
    const o = data, v = o.verdict;
    $("readout").textContent = `Scan ${o.total_ms} ms · model ${o.stage_ms.model} ms`;
    const mid = el("div");
    mid.appendChild(el("p", "advice", v.advice));
    if (v.note) { const n = el("div", "note"); n.appendChild(el("b", "", v.note_label)); n.appendChild(document.createTextNode(v.note)); mid.appendChild(n); }
    if (o.forced) mid.appendChild(el("p", "caveat", "This photo did not pass the usual checks and was read anyway. Treat the result with extra caution."));
    setPage({
      eyebrow: "Result · what to do next", headline: v.headline, lede: v.body, mid, tone: v.tone, mark: true,
      actions: [button("Done", () => render("idle"), true), button("Check another spot", cam === "off" ? pickFile : startLive)],
    });
  }

  if (state === "refused" || state === "error") {
    const o = data, v = o.verdict;
    $("readout").textContent = state === "refused" ? `Stopped after ${o.total_ms} ms` : "Stopped";
    const mid = el("div");
    mid.appendChild(el("p", "advice", v.advice));
    if (v.note) { const n = el("div", "note"); n.appendChild(el("b", "", v.note_label)); n.appendChild(document.createTextNode(v.note)); mid.appendChild(n); }
    const actions = [button("Take another photo", retake, true)];
    if (o.refusal && o.refusal.can_override) actions.push(button("Check it anyway", () => scan(true)));
    else actions.push(button("Back to start", () => render("idle")));
    setPage({ eyebrow: state === "refused" ? "Result · nothing was read" : "Result", headline: v.headline, lede: v.body, mid, tone: v.tone, mark: true, actions });
  }
  renderDrawer();
}

function tickSteps(box) {
  const steps = box.querySelectorAll(".step");
  let i = 0;
  steps[0].classList.add("now");
  S.ticker = setInterval(() => {
    if (body.dataset.state !== "reading") return clearInterval(S.ticker);
    if (i < steps.length - 1) { steps[i].classList.replace("now", "done"); i++; steps[i].classList.add("now"); }
  }, 350);
}

// ---------- camera ----------
function startLive() {
  forgetPhoto();
  if (S.camera === "pi") {
    $("field").src = "/preview.mjpg?t=" + Date.now(); // a fresh URL every time: never reuse a finished stream
    render("live");
  } else if (S.camera === "webcam") {
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false })
      .then((stream) => { S.stream = stream; $("webcam").srcObject = stream; render("live"); })
      .catch(() => { S.camera = "off"; body.dataset.camera = "off"; render("idle"); });
  }
}
function stopWebcam() {
  if (S.stream) { S.stream.getTracks().forEach((t) => t.stop()); S.stream = null; $("webcam").srcObject = null; }
}
async function takePhoto() {
  if (S.camera === "pi") {
    const r = await fetch("/capture", { method: "POST" });
    if (!r.ok) return render("error", { verdict: errorWords((await r.json()).error) });
    showPhoto(await r.blob());
  } else {
    const v = $("webcam"), c = $("canvas"), side = Math.min(v.videoWidth, v.videoHeight);
    c.width = c.height = Math.min(side, 1024);
    c.getContext("2d").drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, c.width, c.height);
    c.toBlob((blob) => { showPhoto(blob); }, "image/jpeg", 0.92);
  }
}
function pickFile() { $("file").value = ""; $("file").click(); }
$("file").addEventListener("change", () => { const f = $("file").files[0]; if (f) showPhoto(f); });

function showPhoto(blob) {
  forgetPhoto();
  S.photoBlob = blob;
  S.photoUrl = URL.createObjectURL(blob);
  $("field").src = S.photoUrl;
  render("captured");
}
function forgetPhoto() {
  if (S.photoUrl) URL.revokeObjectURL(S.photoUrl);
  S.photoUrl = null; S.photoBlob = null; S.outcome = null;
  $("field").removeAttribute("src");
}
function retake() { S.camera === "off" ? (forgetPhoto(), pickFile()) : startLive(); }

// ---------- the scan ----------
async function scan(force) {
  render("reading");
  const form = new FormData();
  // The server remembers the last photo it took or received, so a Pi capture
  // and a forced re-read send nothing; an upload or webcam shot sends the file.
  if (S.photoBlob && !(S.camera === "pi" && S.photoFromPi)) form.append("image", S.photoBlob, "photo.jpg");
  if (force) form.append("force", "1");
  let o;
  try {
    const r = await fetch("/scan", { method: "POST", body: form });
    o = await r.json();
    if (!r.ok) o = { status: "error", verdict: errorWords(o.error), stage_ms: {}, total_ms: 0 };
  } catch (e) {
    o = { status: "error", verdict: errorWords("The scanner did not answer."), stage_ms: {}, total_ms: 0 };
  }
  S.outcome = o;
  await new Promise((r) => setTimeout(r, 700)); // let the checklist finish ticking
  render(o.status === "ok" ? "result" : o.status === "refused" ? "refused" : "error", o);
}
function errorWords(msg) {
  return { state: "error", tone: "info", headline: "THAT SCAN DID NOT FINISH", body: msg || "Something went wrong.", advice: "Take another photo. If this keeps happening, ask a staff member.", note_label: "", note: "" };
}

// ---------- staff drawer ----------
function renderDrawer() {
  const b = $("drawer-body");
  b.replaceChildren();
  const o = S.outcome;
  if (!o) { b.appendChild(el("p", "muted", "No scan yet.")); return; }
  const h = (t) => b.appendChild(el("h3", "", t));
  const kv = (k, v) => { const r = el("div", "kv"); r.appendChild(el("span", "", k)); r.appendChild(el("span", "", v)); b.appendChild(r); };
  if (o.prediction) {
    h("Model output (calibrated)");
    Object.entries(o.prediction.probs_pct).forEach(([k, v]) => {
      const r = el("div", "bar" + (k === o.prediction.label ? " flagged" : ""));
      r.appendChild(el("span", "", k.replace("_", "-")));
      const t = el("div", "bar-track"); const f = el("div", "bar-fill"); f.style.width = v + "%"; t.appendChild(f); r.appendChild(t);
      r.appendChild(el("span", "", v.toFixed(0) + "%")); b.appendChild(r);
    });
    kv("decision", o.prediction.flagged ? "flagged for a doctor" : "not flagged");
    kv("confidence", o.prediction.confidence_pct.toFixed(0) + "%");
    kv("verdict state", o.verdict.state);
    if (o.forced) kv("photo checks", "overridden by visitor");
  } else if (o.refusal) {
    h("Stopped by the photo check"); kv("reason", o.refusal.code); kv("can override", o.refusal.can_override ? "yes" : "no");
  }
  h("What the photo check measured");
  Object.entries(o.measured || {}).forEach(([k, v]) => kv(k.replace(/_/g, " "), Number(v).toFixed(3)));
  h("Timing (ms)");
  Object.entries(o.stage_ms || {}).forEach(([k, v]) => kv(k, String(v)));
  kv("total", String(o.total_ms));
}
$("staff-open").addEventListener("click", () => { renderDrawer(); $("drawer").hidden = false; });
$("staff-close").addEventListener("click", () => { $("drawer").hidden = true; });
$("quit").addEventListener("click", () => {
  const now = Date.now();
  if (now - S.quitArmed > 3000) { S.quitArmed = now; $("quit").textContent = "Tap again to exit"; setTimeout(() => { $("quit").textContent = "Exit kiosk"; }, 3000); return; }
  fetch("/quit", { method: "POST" });
  $("quit").textContent = "Exiting…";
});

// Pi captures already live on the server; mark them so /scan is not sent a copy.
const _showPhoto = showPhoto;
showPhoto = function (blob, fromPi) { S.photoFromPi = !!fromPi; _showPhoto(blob); };
const _takePhoto = takePhoto;
takePhoto = async function () {
  if (S.camera !== "pi") return _takePhoto();
  const r = await fetch("/capture", { method: "POST" });
  if (!r.ok) return render("error", { verdict: errorWords((await r.json()).error) });
  showPhoto(await r.blob(), true);
};

boot();
