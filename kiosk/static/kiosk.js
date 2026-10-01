/* E.P.I.V.U.E. kiosk page. One state machine, one request per scan.

   States:  idle -> live -> captured -> reading -> result | refused | error

   Where the photo comes from:
     "pi"      the server streams /preview.mjpg and takes the photo (/capture)
     "webcam"  the browser's own camera, on a laptop
     "off"     no camera: pick a file

   Every word a visitor reads comes from the server (dermascan/verdict.py). The
   one exception is NO_ANSWER below: when the server cannot be reached at all,
   it cannot send words either. */

const $ = (id) => document.getElementById(id);
const body = document.body;
const S = { camera: "off", photoBlob: null, photoUrl: null, photoOnServer: false, outcome: null, stream: null,
            exitNeedsCode: false, exitArmedAt: 0, code: "" };

const NO_ANSWER = {
  status: "error", stage_ms: {}, total_ms: 0, measured: {}, refusal: null, caveat: "",
  verdict: { state: "error", tone: "info", headline: "THAT SCAN DID NOT FINISH",
             body: "The scanner did not answer.", advice: "Take another photo. If this keeps happening, ask a staff member.",
             note_label: "", note: "" },
};

// fetch + JSON, never throws: a dead server or a non-JSON reply becomes NO_ANSWER.
async function postJSON(url, form) {
  try {
    const r = await fetch(url, { method: "POST", body: form });
    return await r.json();
  } catch (e) {
    return NO_ANSWER;
  }
}

// ---------- boot ----------
async function boot() {
  try {
    const h = await (await fetch("/health")).json();
    S.camera = h.camera ? "pi" : (navigator.mediaDevices && navigator.mediaDevices.getUserMedia ? "webcam" : "off");
    S.exitNeedsCode = !!h.exit_needs_code;
    $("model-line").textContent = h.ok ? `${h.model.file} · ${h.model.tta ? "4 views" : "1 view"}` : "model not loaded";
  } catch (e) {
    S.camera = "off";
  }
  body.dataset.camera = S.camera;
  render("idle");
}

// ---------- small DOM helpers ----------
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
  const box = el("div");
  items.forEach((t, i) => {
    const r = el("div", "row");
    if (numbered) r.appendChild(el("span", "row-num", String(i + 1)));
    r.appendChild(el("span", "", t));
    box.appendChild(r);
  });
  return box;
}
function verdictBlock(o) {
  const v = o.verdict, box = el("div");
  box.appendChild(el("p", "advice", v.advice));
  if (v.note) {
    const n = el("div", "note");
    n.appendChild(el("b", "", v.note_label));
    n.appendChild(document.createTextNode(v.note));
    box.appendChild(n);
  }
  if (o.caveat) box.appendChild(el("p", "caveat", o.caveat));
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

// ---------- the screens ----------
function render(state, o) {
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
    setPage({
      eyebrow: "Reading",
      headline: "Reading the spot…",
      lede: "Everything happens on this device. No internet needed.",
      mid: rows(["Checks the photo shows one spot on skin", "Compares the spot with the on-device model", "Puts what to do next into plain words"]),
      actions: [],
    });
  }

  if (state === "result") {
    $("readout").textContent = `Scan ${o.total_ms} ms · model ${o.stage_ms.model} ms`;
    setPage({
      eyebrow: "Result · what to do next", headline: o.verdict.headline, lede: o.verdict.body,
      mid: verdictBlock(o), tone: o.verdict.tone, mark: true,
      actions: [button("Done", () => render("idle"), true), button("Check another spot", cam === "off" ? pickFile : startLive)],
    });
  }

  if (state === "refused" || state === "error") {
    $("readout").textContent = state === "refused" ? `Stopped after ${o.total_ms} ms` : "Stopped";
    const actions = [button("Take another photo", retake, true)];
    if (o.refusal && o.refusal.can_override) actions.push(button("Check it anyway", () => scan(true)));
    else actions.push(button("Back to start", () => render("idle")));
    setPage({
      eyebrow: state === "refused" ? "Result · nothing was read" : "Result",
      headline: o.verdict.headline, lede: o.verdict.body, mid: verdictBlock(o), tone: o.verdict.tone, mark: true, actions,
    });
  }
  renderDrawer();
}

// ---------- camera and photo ----------
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
    let r;
    try { r = await fetch("/capture", { method: "POST" }); } catch (e) { return render("error", NO_ANSWER); }
    if (!r.ok) return render("error", await r.json().catch(() => NO_ANSWER));
    return showPhoto(await r.blob(), true); // the server already holds this photo
  }
  const v = $("webcam"), c = $("canvas"), side = Math.min(v.videoWidth, v.videoHeight);
  c.width = c.height = Math.min(side, 1024);
  c.getContext("2d").drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, c.width, c.height);
  c.toBlob((blob) => showPhoto(blob, false), "image/jpeg", 0.92);
}
function pickFile() { $("file").value = ""; $("file").click(); }
$("file").addEventListener("change", () => { const f = $("file").files[0]; if (f) showPhoto(f, false); });

function showPhoto(blob, onServer) {
  forgetPhoto();
  S.photoBlob = blob;
  S.photoOnServer = onServer;
  S.photoUrl = URL.createObjectURL(blob);
  $("field").src = S.photoUrl;
  render("captured");
}
function forgetPhoto() {
  if (S.photoUrl) URL.revokeObjectURL(S.photoUrl);
  if (S.photoBlob || S.outcome) fetch("/forget", { method: "POST" }).catch(() => {}); // the server drops its copy too
  S.photoUrl = null; S.photoBlob = null; S.photoOnServer = false; S.outcome = null;
  $("field").removeAttribute("src");
}
function retake() { if (S.camera === "off") { forgetPhoto(); pickFile(); } else startLive(); }

// ---------- the scan ----------
async function scan(force) {
  render("reading");
  const form = new FormData();
  // A Pi capture is already on the server, and so is any photo being re-read with
  // force; only a first look at an upload or a webcam shot sends the file.
  if (S.photoBlob && !S.photoOnServer) { form.append("image", S.photoBlob, "photo.jpg"); S.photoOnServer = true; }
  if (force) form.append("force", "1");
  const o = await postJSON("/scan", form);
  S.outcome = o;
  render(o.status === "ok" ? "result" : o.status === "refused" ? "refused" : "error", o);
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
      const t = el("div", "bar-track"), f = el("div", "bar-fill");
      f.style.width = v + "%"; t.appendChild(f); r.appendChild(t);
      r.appendChild(el("span", "", v.toFixed(0) + "%"));
      b.appendChild(r);
    });
    kv("decision", o.prediction.flagged ? "flagged for a doctor" : "not flagged");
    kv("confidence", o.prediction.confidence_pct.toFixed(0) + "%");
    kv("verdict state", o.verdict.state);
    if (o.forced) kv("photo checks", "overridden by visitor");
  } else if (o.refusal) {
    h("Stopped by the photo check");
    kv("reason", o.refusal.code);
    kv("can override", o.refusal.can_override ? "yes" : "no");
  }
  if (Object.keys(o.measured || {}).length) {
    h("What the photo check measured");
    Object.entries(o.measured).forEach(([k, v]) => kv(k.replace(/_/g, " "), Number(v).toFixed(3)));
  }
  if (Object.keys(o.stage_ms || {}).length) {
    h("Timing (ms)");
    Object.entries(o.stage_ms).forEach(([k, v]) => kv(k, String(v)));
    kv("total", String(o.total_ms));
  }
}
$("staff-open").addEventListener("click", () => { renderDrawer(); closeKeypad(); $("drawer").hidden = false; });
$("staff-close").addEventListener("click", () => { $("drawer").hidden = true; closeKeypad(); });

// Exit: two taps, or the staff code on a keypad when one is set (config.staff_passcode).
$("quit").addEventListener("click", () => {
  if (S.exitNeedsCode) return openKeypad();
  const now = Date.now();
  if (now - S.exitArmedAt > 3000) {
    S.exitArmedAt = now;
    $("quit").textContent = "Tap again to exit";
    setTimeout(() => { $("quit").textContent = "Exit kiosk"; }, 3000);
    return;
  }
  postJSON("/quit", new FormData());
  $("quit").textContent = "Exiting…";
});
function openKeypad() {
  S.code = "";
  const pad = $("keypad");
  pad.replaceChildren();
  const shown = el("div", "code-dots", "· · · ·");
  pad.appendChild(shown);
  const grid = el("div", "keys");
  ["1", "2", "3", "4", "5", "6", "7", "8", "9", "Clear", "0", "Cancel"].forEach((k) => {
    grid.appendChild(button(k, () => keypadPress(k, shown)));
  });
  pad.appendChild(grid);
  pad.hidden = false;
}
function closeKeypad() { $("keypad").hidden = true; S.code = ""; }
async function keypadPress(k, shown) {
  if (k === "Cancel") return closeKeypad();
  if (k === "Clear") S.code = "";
  else if (S.code.length < 4) S.code += k;
  shown.textContent = S.code ? "●".repeat(S.code.length) + " ·".repeat(4 - S.code.length) : "· · · ·";
  if (S.code.length === 4) {
    const form = new FormData();
    form.append("code", S.code);
    S.code = "";
    const r = await postJSON("/quit", form);
    shown.textContent = r.ok ? "Exiting…" : "Wrong code";
  }
}

boot();
