/* E.P.I.V.U.E. kiosk page. One state object, one render function per screen.

   Screens (the nav keys and the flow):
     home -> frame -> check -> reading -> result
     saved · questions · settings          (nav keys, any time)

   Where the photo comes from:
     "pi"      the server streams /preview.mjpg and takes the photo (/capture)
     "webcam"  the browser's own camera, on a laptop or phone
     "off"     no camera: choose a picture

   Every verdict, photo reading and answer a visitor reads comes from the server
   (dermascan/verdict.py, dermascan/answers.py). This file holds only the screen
   chrome from the design (titles, steps, button labels) and the two messages for
   when the server cannot be reached and so cannot send words: NO_ANSWER and
   NO_REPLY. */

const $ = (id) => document.getElementById(id);
const app = $("app");

const S = {
  screen: "home", camera: "off", source: "camera", stream: null,
  photoBlob: null, photoUrl: null, photoOnServer: false,
  check: null, outcome: null, viewing: null, // viewing: a saved scan {id, site} shown instead of the live one
  savedAs: null, savedId: null, history: null, thread: [], suggestions: [],
  spots: [], sites: [], filter: null, stats: { count: 0, bytes: 0 },
  health: null, settings: { allow_read_anyway: true, show_staff_details: true },
  staffCode: null, code: "", eraseArmedAt: 0, scanAbort: null,
};

const NO_ANSWER = {
  status: "error", stage_ms: {}, total_ms: 0, measured: {}, refusal: null, caveat: "", signs: null, sign_lines: [],
  verdict: { state: "error", tone: "info", chip: "NOT READ", headline: "THAT SCAN DID NOT FINISH",
             body: "The scanner did not answer.", advice: "Take another photo. If this keeps happening, ask a staff member.",
             note_label: "", note: "" },
};

const NO_REPLY = { text: "The kiosk did not answer. Ask a staff member.", badge: "", reviewed: false };

// Arc and line colours for the A B C D E signs, by tier (null = not measured).
const ARC = { 0: "#8C9BB0", 1: "#C79A6B", 2: "#E0645A", null: "#7E9BD0" };
const LINE = { 0: "#8B95A5", 1: "#7A4B2A", 2: "#C0362C" };
const RING = { neutral: "#5B6675", warning: "#7A4B2A", urgent: "#C0362C", info: "#7E9BD0" };
const HISTORY_ARC = "#3DDBD9"; // E has an earlier photo of this spot to compare with
const THUMB = { urgent: "#E0645A", warning: "#C79A6B" }; // saved-scan rings; anything else is plain grey
const PROB_NAME = { benign: "Benign", pre_cancerous: "Pre-cancerous", malignant: "Malignant" }; // staff only
const SIGN_NAME = { A: "Asymmetry", B: "Border", C: "Colours", D: "Size in frame", E: "Evolving" };
const PILL = { 0: "NORMAL", 1: "BORDERLINE", 2: "STANDS OUT" };

// ---------- network: never throws ----------
async function post(url, body, opts = {}) {
  try {
    const isJson = body && !(body instanceof FormData);
    const r = await fetch(url, {
      method: "POST", signal: opts.signal,
      headers: isJson ? { "Content-Type": "application/json" } : undefined,
      body: isJson ? JSON.stringify(body) : body,
    });
    const data = await r.json().catch(() => null);
    return { ok: r.ok, status: r.status, data };
  } catch (e) {
    return { ok: false, status: 0, data: null, aborted: e && e.name === "AbortError" };
  }
}
async function getJSON(url) {
  try { const r = await fetch(url, { cache: "no-store" }); return r.ok ? await r.json() : null; } catch (e) { return null; }
}
function form(fields) {
  const f = new FormData();
  Object.entries(fields).forEach(([k, v]) => { if (v !== undefined && v !== null) f.append(k, v); });
  return f;
}
function staffForm(fields = {}) { return form({ ...fields, code: S.staffCode || undefined }); }

// ---------- tiny DOM helpers ----------
function h(tag, cls, ...kids) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  kids.flat().forEach((k) => { if (k !== null && k !== undefined && k !== false) n.append(k.nodeType ? k : String(k)); });
  return n;
}
function btn(label, onClick, cls = "btn") {
  const b = h("button", cls, label);
  b.type = "button";
  b.addEventListener("click", onClick);
  return b;
}
const eyebrow = (t) => h("p", "eyebrow", t);
const h1 = (t, cls = "") => h("h1", "h1 " + cls, t);
const lede = (t, cls = "") => (t ? h("p", "lede " + cls, t) : null);
function actions(kind, ...buttons) { return h("div", "actions " + (kind || ""), buttons.filter(Boolean)); }
function eyebrowWithChip(text, chipText) {
  return h("div", "eyebrow-row", eyebrow(text), chipText ? h("span", "chip", chipText) : null);
}

// ---------- the clock ----------
function tick() {
  const d = new Date();
  const t = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }).replace(/\s/g, "");
  const day = `${String(d.getDate()).padStart(2, "0")} ${d.toLocaleDateString("en", { month: "short" })}`;
  $("clock").textContent = `${t} · ${day}`;
}
function timeOf(epochSeconds) {
  return new Date(epochSeconds * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

// ---------- the instrument: aperture and caption ----------
function shown() { return S.viewing ? S.viewing.outcome : S.outcome; }

function paintInstrument() {
  const d = app.dataset;
  d.screen = S.screen;
  d.camera = S.camera;
  d.source = S.source;
  const o = shown();
  const onResult = (S.screen === "result" || S.screen === "questions") && o;
  const hasPhoto = $("field").getAttribute("src") && (["check", "reading", "result", "questions", "saved"].includes(S.screen));
  toggle(d, "photo", hasPhoto && !(S.screen === "questions" && !o));
  toggle(d, "verdict", onResult && o.status === "ok");
  toggle(d, "signs", onResult && o.status === "ok" && o.signs);
  d.tone = onResult || (S.screen === "check" && S.check && S.check.status !== "pass") ? (o && onResult ? o.verdict.tone : "info") : "";
  if (onResult && o.status === "ok") {
    document.querySelector(".verdict-ring").setAttribute("stroke", RING[o.verdict.tone] || RING.neutral);
    const tiers = Object.fromEntries((o.signs || []).map((s) => [s.letter, s.tier]));
    $("arcs").querySelectorAll("path").forEach((p) => { p.style.stroke = ARC[tiers[p.dataset.sign] ?? null]; });
    const hist = currentHistory();
    if (hist && hist.count > 1) $("arcs").querySelector('[data-sign="E"]').style.stroke = HISTORY_ARC;
  }
  $("ap-rest").textContent = S.screen === "frame" && S.camera === "off"
    ? "No camera here — choose a picture" : "Put the spot inside the ring";
  $("cap-standby").textContent = !S.health || !S.health.ok ? "THE SCANNER IS NOT READY — ASK STAFF"
    : S.camera === "off" && ["home", "frame"].includes(S.screen) ? "READY · CHOOSE A PICTURE TO TRY IT" : "READY · ON-DEVICE MODEL";
  document.querySelectorAll(".nav button").forEach((b) => {
    const mine = b.dataset.go === S.screen || (b.dataset.go === "frame" && ["check", "reading", "result"].includes(S.screen) && !S.viewing);
    if (mine) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  });
}
function toggle(d, key, on) { if (on) d[key] = ""; else delete d[key]; }

function showPhotoUrl(url) {
  if (url) $("field").src = url; else $("field").removeAttribute("src");
}

// ---------- navigation ----------
function go(screen) {
  if (S.scanAbort) { S.scanAbort.abort(); S.scanAbort = null; }
  closeOverlays();
  const from = S.screen;
  if (screen !== "frame") stopWebcam();
  if (S.viewing && !["result", "questions"].includes(screen)) S.viewing = null;
  // Leaving the live view (stops the Pi preview stream), Saved, or a saved scan puts
  // this visitor's own photo, if any, back in the aperture.
  if (!S.viewing && (from === "frame" || from === "saved" || from === "result" || from === "questions") && screen !== "frame") showPhotoUrl(S.photoUrl);
  // The staff area locks itself as soon as staff leave Settings.
  if (from === "settings" && screen !== "settings") S.staffCode = null;
  S.screen = screen;
  RENDER[screen]();
  paintInstrument();
  $("screen").scrollTop = 0;
}
document.querySelectorAll(".nav button").forEach((b) => b.addEventListener("click", () => {
  const to = b.dataset.go;
  if (to === "frame") return startCheck();
  if (to === "home") return finish();
  go(to);
}));

function setScreen(...nodes) { $("screen").replaceChildren(...nodes.filter(Boolean)); }

// ---------- screens ----------
const RENDER = {
  home() {
    setScreen(
      eyebrow("SKIN CHECK · TAKES ABOUT A MINUTE"),
      h1("Check a spot on your skin"),
      lede("Nothing you photograph leaves this device. No name, no account."),
      h("div", "mid", ["Fill the ring with the spot and hold still", "Tap once to take the photo", "Read what to do next, in plain words"]
        .map((t, i) => h("div", "row", h("span", "num", i + 1), h("span", "", t)))),
      actions("", btn("Start a skin check", startCheck, "btn primary"), btn("Saved spots", () => go("saved"))),
    );
  },

  frame() {
    const off = S.camera === "off";
    setScreen(
      eyebrow("STEP 1 OF 3 · FRAME THE SPOT"),
      h1(off ? "Choose a photo of the spot" : "Fill the ring, then hold still"),
      lede(off ? "This computer has no camera. Pick a close, sharp photo of one spot on skin."
               : "Rest the camera on the skin so the spot sits in the middle of the ring."),
      h("div", "mid", ["Spot inside the ring, edges included", "Even light — no shadow, no glare", off ? "One spot, close up and in focus" : "Two seconds still, then tap"]
        .map((t) => h("div", "row", h("span", "tick"), t))),
      off ? actions("one", btn("Choose a picture", pickFile, "btn primary"))
          : actions("", btn("Take the photo", takePhoto, "btn primary"), btn("Choose a picture", pickFile)),
    );
  },

  check() {
    const c = S.check;
    if (!c) {
      setScreen(eyebrow("STEP 2 OF 3 · CHECK THE PHOTO"), h1("Checking the photo…"), lede("Light, focus, and whether there is one spot to read."),
        readingsBlock(null), actions("", btn("Check this spot", () => {}, "btn primary"), btn("Take another", retake)));
      $("screen").querySelector(".btn.primary").disabled = true;
      return;
    }
    const pass = c.status === "pass";
    const canForce = c.refusal && c.refusal.can_override;
    setScreen(
      eyebrow(pass ? "STEP 2 OF 3 · CHECK THE PHOTO" : "STEP 2 OF 3 · THIS PHOTO CANNOT BE READ"),
      h1(c.headline, pass ? "" : "verdict"),
      lede(pass && S.source === "import" ? "Read from the picture you chose — " + c.lede.charAt(0).toLowerCase() + c.lede.slice(1) : c.lede),
      !pass && c.verdict ? h("p", "advice", c.verdict.advice) : null,
      readingsBlock(c.readings),
      pass ? actions("", btn("Check this spot", () => scan(false), "btn primary"), btn("Take another", retake))
           : actions(canForce ? "" : "one", btn("Take another", retake, "btn primary"), canForce ? btn("Check it anyway", () => scan(true)) : null),
    );
  },

  reading() {
    // Honest steps: the photo check already ran (/check); the rest is one request.
    const steps = [["Found the spot and drew its outline", "done"], ["Comparing with the on-device model", ""], ["Measuring the warning signs", ""]]
      .map(([t, state]) => h("div", "step " + state, h("i"), t));
    setScreen(
      eyebrow("STEP 3 OF 3 · READING"),
      h1("Reading the spot…"),
      h("div", "progress", h("i")),
      h("div", "steps", steps),
      h("p", "note-small", "Everything happens on this device. No internet needed."),
      actions("one", btn("Cancel", retake, "btn")),
    );
  },

  result() {
    const o = shown();
    if (!o) return go("home");
    const v = o.verdict, ok = o.status === "ok";
    const lines = (o.sign_lines || []).map((l) => h("div", "sign", h("i", "", ""), h("span", "", l.text)));
    lines.forEach((n, i) => { n.firstChild.style.background = LINE[o.sign_lines[i].tier]; });
    const saving = S.viewing ? null : saveButton(o);
    // E: a spot saved more than once gets "Compare" - first photo beside this one.
    const hist = currentHistory();
    const compare = hist && hist.count > 1 ? btn(`Compare · ${hist.count} scans`, openCompare, "soft history") : null;
    const staff = S.settings.show_staff_details && ok ? btn(compare ? "Staff →" : "Details for staff →", openStaff, "soft outline") : null;
    setScreen(
      eyebrowWithChip(S.viewing ? `SAVED · ${(S.viewing.group || S.viewing.site).toUpperCase()} · ${timeOf(S.viewing.saved_at)}` : "RESULT · WHAT TO DO NEXT", v.chip),
      h1(v.headline, "verdict"),
      h("div", "tone-bar"),
      lede(v.body, "tight"),
      h("p", "advice", v.advice),
      o.caveat ? h("p", "caveat", o.caveat) : null,
      lines.length ? h("div", "saw", h("div", "label", "WHAT THE SCAN SAW"), lines) : null,
      v.note ? h("div", "notebox", h("div", "label", v.note_label), h("p", "", v.note)) : null,
      ok ? actions(lines.length ? "even flush" : "even",
                   S.viewing ? btn("Back to saved", () => go("saved"), "btn compact primary") : btn("Done", finish, "btn compact primary"),
                   S.viewing ? btn("Delete this scan", deleteViewed, "btn compact danger") : saving)
         : actions("even", btn("Take another photo", retake, "btn compact primary"), btn("Back to start", finish, "btn compact")),
      ok ? h("div", "subrow", compare, btn(compare ? "Ask a question" : "Ask a question about this", () => go("questions"), "soft"), staff) : null,
    );
  },

  saved() {
    const all = S.spots;
    const total = all.reduce((n, s) => n + s.scans.length, 0);
    const spots = S.filter ? all.filter((s) => s.group === S.filter) : all;
    const filters = h("div", "filters",
      filterChip(`All scans · ${total}`, null),
      all.map((s) => filterChip(`${s.group} · ${s.scans.length}`, s.group)));
    const list = h("div", "list", spots.map((s) => {
      const latest = s.scans[0];
      const thumb = h("span", "thumb");
      thumb.style.backgroundImage = `url(/saved/${latest.id}.jpg)`;
      thumb.style.boxShadow = `0 0 0 2px ${THUMB[latest.tone] || "#C6CDD9"}`;
      const row = h("button", "spot", thumb,
        h("span", "spot-t", h("b", "", s.group), h("span", "", s.scans.length > 1
          ? `${s.scans.length} scans · first ${timeOf(s.scans[s.scans.length - 1].saved_at)} · last ${timeOf(latest.saved_at)}`
          : `1 scan · ${timeOf(latest.saved_at)}`)),
        h("span", "pill " + (latest.tone === "neutral" ? "" : latest.tone), latest.headline));
      row.type = "button";
      row.addEventListener("click", () => openSaved(latest.id));
      return row;
    }));
    const latest = all.length ? all[0].scans[0] : null;
    setScreen(
      eyebrow("SAVED SPOTS · UNTIL THE EVENT ENDS"),
      h1("Saved spots"),
      total ? filters : null,
      total ? list : h("p", "empty", "Nothing saved yet. After a check, tap “Save this scan” to keep it here until staff end the event. It is never written to the kiosk’s card."),
      actions("even", btn("Add a new check", startCheck, "btn compact primary"),
              latest ? btn("Open latest scan", () => openSaved(latest.id), "btn compact") : null),
    );
    // The instrument shows the newest scan of the spot in view.
    const focus = spots[0];
    if (focus) {
      showPhotoUrl(`/saved/${focus.scans[0].id}.jpg`);
      $("cap-saved-title").textContent = `${focus.group.toUpperCase()} · ${focus.scans.length} SCAN${focus.scans.length > 1 ? "S" : ""}`;
      const dots = [];
      focus.scans.slice(0, 5).reverse().forEach((sc, i, arr) => {
        if (i) dots.push(h("b"));
        dots.push(h("i", i === arr.length - 1 ? "now" : ""));
      });
      $("cap-dots").replaceChildren(...dots);
    } else showPhotoUrl(null);
  },

  questions() {
    const o = shown();
    const v = o && o.verdict;
    const thread = h("div", "thread");
    if (!S.thread.length) {
      thread.append(h("div", "label", "ASK ANYTHING ABOUT A SKIN CHECK"),
        h("div", "qa-a", "Tap a question below" + (S.camera === "pi" ? "." : ", or type your own.") +
          " Every answer is written out in advance by the project team, so nothing is made up on the spot."));
    }
    S.thread.forEach((qa) => {
      thread.append(h("div", "qa",
        h("div", "label", "YOU ASKED"), h("div", "qa-q", qa.question),
        h("div", "qa-a", qa.pending ? "…" : qa.text),
        qa.pending || !qa.badge ? null : h("div", "badge" + (qa.reviewed ? " ok" : ""), h("i"), qa.badge)));
    });
    const input = h("input");
    input.type = "text"; input.placeholder = "Type a question"; input.maxLength = 200; input.enterKeyHint = "send"; // 200 = config.QUESTION_MAX_CHARS
    const sendTyped = () => { const q = input.value.trim(); if (q) { input.value = ""; ask(q); } };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") sendTyped(); });
    setScreen(
      eyebrowWithChip("QUESTIONS · WRITTEN ANSWERS", v ? v.chip : "NO SCAN YET"),
      h1(v ? "Ask about your result" : "Ask a question", "small"),
      thread,
      h("div", "label push", S.thread.length ? "OR TAP ANOTHER QUESTION" : "TAP A QUESTION"),
      h("div", "suggest", S.suggestions.map((q) => btn(q, () => ask(q), "q"))),
      h("div", "ask", input, btn("Ask", sendTyped, "btn primary")),
      actions("one flush", o ? btn("Back to the result", () => go("result"), "btn compact") : btn("Start a skin check", startCheck, "btn compact")),
    );
    thread.scrollTop = thread.scrollHeight;
  },

  settings() {
    if (locked()) return renderKeypad();
    const t = (key, title, desc) => {
      const row = h("button", "setting", h("span", "setting-t", h("b", "", title), h("span", "", desc)), h("span", "toggle", h("i")));
      row.type = "button";
      row.setAttribute("role", "switch");
      row.setAttribute("aria-checked", String(on(key)));
      row.addEventListener("click", () => changeSetting(key, !S.settings[key]));
      return row;
    };
    // The design's wording is "refuse", the server's switch is "read anyway": the same switch, flipped.
    const on = (key) => (key === "allow_read_anyway" ? !S.settings[key] : !!S.settings[key]);
    const m = S.health && S.health.model;
    const erase = btn("End event — erase all", eraseAll, "btn compact danger");
    setScreen(
      eyebrow("SETTINGS · STAFF"),
      h1("Settings"),
      h("div", "mid",
        t("allow_read_anyway", "Refuse photos that are not clear", "Off: an unclear photo can still be read, with a note. On: it is refused and the visitor takes another."),
        t("show_staff_details", "Details for staff under each result", "Numbers and timings behind one button. Visitors see them only if they tap it."),
        h("div", "fact", h("span", "", "Saved scans, in memory only"), h("b", "", `${S.stats.count} scan${S.stats.count === 1 ? "" : "s"} · ${size(S.stats.bytes)}`)),
        h("div", "fact", h("span", "", "Scanner"), h("b", "", m ? `${m.file} · ${m.tta ? "4 views" : "1 view"}` : "not loaded")),
        h("div", "fact", h("span", "", "Close the kiosk app"), btn("Exit kiosk", quitKiosk, "linkbtn")),
      ),
      actions("even", btn("Lock the staff area", () => { S.staffCode = null; go("home"); }, "btn compact"), erase),
    );
  },
};

function size(bytes) { return bytes < 1e6 ? `${Math.round(bytes / 1e3)} KB` : `${(bytes / 1e6).toFixed(1)} MB`; }

function readingsBlock(readings) {
  const rows = (readings || [{ name: "Light" }, { name: "Focus" }, { name: "Spot in frame" }]).map((r, i, arr) => {
    const level = r.level ?? 0;
    const bars = h("span", "bars" + (r.level === 2 ? " warn" : r.level !== undefined && r.level !== null && r.level < 2 ? " bad" : ""),
      [0, 1, 2].map((k) => h("i", k < level ? "on" : "")));
    return h("div", "row" + (i === arr.length - 1 ? " ruled" : ""), h("span", "reading-name", r.name), bars, h("span", "reading-word", r.word || "…"));
  });
  return h("div", "mid", rows);
}

function filterChip(label, group) {
  const b = btn(label, () => { S.filter = group; go("saved"); }, "filter");
  b.setAttribute("aria-pressed", String(S.filter === group));
  return b;
}

function saveButton(o) {
  if (S.savedAs) { const b = btn(`Saved · ${S.savedAs}`, () => go("saved"), "btn compact"); return b; }
  return btn("Save this scan", openSiteSheet, "btn compact");
}

// ---------- camera and photo ----------
function startCheck() {
  forgetPhoto();
  S.source = "camera";
  if (S.camera === "pi") {
    $("field").src = "/preview.mjpg?t=" + Date.now(); // a fresh URL every time: never reuse a finished stream
    go("frame");
  } else if (S.camera === "webcam") {
    go("frame");
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false })
      .then((stream) => {
        if (S.screen !== "frame") { stream.getTracks().forEach((t) => t.stop()); return; }
        S.stream = stream; $("webcam").srcObject = stream;
      })
      .catch(() => { S.camera = "off"; go("frame"); });
  } else go("frame");
}
function stopWebcam() {
  if (S.stream) { S.stream.getTracks().forEach((t) => t.stop()); S.stream = null; $("webcam").srcObject = null; }
}
async function takePhoto() {
  if (S.camera === "pi") {
    let r;
    try { r = await fetch("/capture", { method: "POST" }); } catch (e) { return showOutcome(NO_ANSWER); }
    if (!r.ok) return showOutcome(await r.json().catch(() => NO_ANSWER));
    return usePhoto(await r.blob(), true, "camera"); // the server already holds this photo
  }
  const v = $("webcam");
  if (!v.videoWidth) return;
  const c = $("canvas"), side = Math.min(v.videoWidth, v.videoHeight);
  c.width = c.height = Math.min(side, 1024);
  c.getContext("2d").drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, c.width, c.height);
  c.toBlob((blob) => usePhoto(blob, false, "camera"), "image/jpeg", 0.92);
}
function pickFile() { $("file").value = ""; $("file").click(); }
$("file").addEventListener("change", () => { const f = $("file").files[0]; if (f) usePhoto(f, false, "import"); });

async function usePhoto(blob, onServer, source) {
  forgetPhoto();
  stopWebcam();
  S.photoBlob = blob; S.photoOnServer = onServer; S.source = source;
  S.photoUrl = URL.createObjectURL(blob);
  showPhotoUrl(S.photoUrl);
  go("check");
  // The photo check alone: milliseconds, no model. It also hands the server the photo.
  const f = new FormData();
  if (!onServer) f.append("image", blob, "photo.jpg");
  const r = await post("/check", f);
  if (S.photoBlob !== blob) return; // another photo replaced this one meanwhile
  if (!r.data) return showOutcome(NO_ANSWER);
  S.photoOnServer = true;
  S.check = r.data;
  if (S.screen === "check") go("check");
}
function forgetPhoto() {
  if (S.photoUrl) URL.revokeObjectURL(S.photoUrl);
  if (S.photoBlob || S.outcome) fetch("/forget", { method: "POST" }).catch(() => {}); // the server drops its copy too
  Object.assign(S, { photoUrl: null, photoBlob: null, photoOnServer: false, outcome: null, check: null, savedAs: null, thread: [] });
  if (!S.viewing) showPhotoUrl(null);
}
function retake() { if (S.camera === "off") { forgetPhoto(); go("frame"); pickFile(); } else startCheck(); }
function finish() { forgetPhoto(); S.viewing = null; loadSuggestions(); go("home"); }

// ---------- the scan ----------
async function scan(force) {
  go("reading");
  const f = new FormData();
  if (S.photoBlob && !S.photoOnServer) { f.append("image", S.photoBlob, "photo.jpg"); S.photoOnServer = true; }
  if (force) f.append("force", "1");
  const ctrl = new AbortController();
  S.scanAbort = ctrl;
  const r = await post("/scan", f, { signal: ctrl.signal });
  if (r.aborted || S.scanAbort !== ctrl) return;
  S.scanAbort = null;
  showOutcome(r.data || NO_ANSWER);
}
function showOutcome(o) {
  S.outcome = o; S.viewing = null; S.thread = []; S.savedAs = null; S.savedId = null; S.history = null;
  loadSuggestions();
  go("result");
}

// ---------- saved scans ----------
async function loadSaved() {
  const d = await getJSON("/saved");
  if (d) { S.spots = d.spots; S.sites = d.sites; S.stats = d.stats; }
  if (S.filter && !S.spots.some((s) => s.group === S.filter)) S.filter = null;
}
const _goSaved = RENDER.saved;
RENDER.saved = function () { _goSaved(); loadSaved().then(() => { if (S.screen === "saved") { _goSaved(); paintInstrument(); } }); };
const _goSettings = RENDER.settings;
const locked = () => S.health && S.health.exit_needs_code && !S.staffCode;
RENDER.settings = function () { _goSettings(); if (!locked()) loadSaved().then(() => { if (S.screen === "settings" && !locked()) _goSettings(); }); };

async function openSaved(id) {
  const d = await getJSON(`/saved/${id}`);
  if (!d) return go("saved");
  S.viewing = { ...d.scan, outcome: d.outcome, history: d.history };
  S.thread = [];
  go("result");
  showPhotoUrl(`/saved/${id}.jpg`);
  paintInstrument();
  loadSuggestions();
}

function openSiteSheet() {
  const sheet = $("sheet");
  const sites = S.sites.length ? S.sites : ["Face", "Neck", "Chest", "Back", "Left arm", "Right arm", "Left leg", "Right leg", "Other"];
  sheet.replaceChildren(
    eyebrow("SAVE THIS SCAN · KEPT UNTIL THE EVENT ENDS"),
    h1("Where is the spot?", "small"),
    lede("Saved scans stay in this kiosk’s memory only, with no name. Staff erase them at the end of the event."),
    h("div", "sites", sites.map((s) => btn(s, () => pickGroup(s)))),
    actions("one", btn("Cancel", closeOverlays)),
  );
  sheet.hidden = false;
}
// Step two of saving: which group (which spot) on this body site. A group's
// history starts at the first scan saved into it.
async function pickGroup(site) {
  await loadSaved(); // groups may have been deleted or erased since this list was last read
  const groups = S.spots.filter((g) => g.site === site);
  if (!groups.length) return saveAs(site, null);
  const rows = groups.map((g) => {
    const thumb = h("span", "thumb");
    thumb.style.backgroundImage = `url(/saved/${g.scans[0].id}.jpg)`;
    const row = h("button", "spot", thumb,
      h("span", "spot-t", h("b", "", g.group), h("span", "", `${g.scans.length} scan${g.scans.length > 1 ? "s" : ""} · first ${timeOf(g.scans[g.scans.length - 1].saved_at)}`)),
      h("span", "pill", "ADD HERE"));
    row.type = "button";
    row.addEventListener("click", () => saveAs(site, g.group));
    return row;
  });
  $("sheet").replaceChildren(
    eyebrow(`SAVE THIS SCAN · ${site.toUpperCase()}`),
    h1("Same spot as before?", "small"),
    lede("Add it to a spot you saved earlier to build its history, or start a new group for a different spot."),
    h("div", "list", rows),
    actions("even", btn("Start a new group", () => saveAs(site, null), "btn compact primary"), btn("Cancel", closeOverlays, "btn compact")),
  );
}
async function saveAs(site, group) {
  const r = await post("/saved", form({ site, group }));
  if (!r.ok) {
    $("sheet").replaceChildren(eyebrow("SAVE THIS SCAN"), h1("That did not save", "small"),
      lede("The group may have just been erased. Try again."), actions("one", btn("Try again", openSiteSheet, "btn compact primary")));
    return;
  }
  closeOverlays();
  S.savedAs = r.data.scan.group; S.savedId = r.data.scan.id; S.history = r.data.history; loadSaved();
  if (S.screen === "result") { RENDER.result(); paintInstrument(); }
}
function currentHistory() { return S.viewing ? S.viewing.history : S.history; }
// After a delete or an erase, the live result's history (or the saved scan itself) may be gone.
async function refreshLiveHistory() {
  if (!S.savedId) return;
  const d = await getJSON(`/saved/${S.savedId}`);
  if (d) S.history = d.history;
  else { S.savedId = null; S.savedAs = null; S.history = null; }
}

// E, shown honestly: the group's first photo beside this one, for a person to compare.
function openCompare() {
  const hist = currentHistory();
  if (!hist) return;
  // Opening the group's first scan compares it with the latest, never with itself.
  const viewingFirst = S.viewing && S.viewing.id === hist.first.id;
  const other = S.viewing && !viewingFirst ? { id: S.viewing.id, at: S.viewing.saved_at, n: hist.number } : { id: hist.latest.id, at: hist.latest.saved_at, n: hist.count };
  const pane = (id, label, at) => {
    const img = h("img", "cmp-img"); img.src = `/saved/${id}.jpg`; img.alt = label;
    return h("figure", "cmp", img, h("figcaption", "", h("b", "", label), ` · ${timeOf(at)}`));
  };
  $("sheet").replaceChildren(
    eyebrow(`E · HOW THIS SPOT LOOKS OVER TIME · ${hist.group.toUpperCase()}`),
    h1(`First check and check ${other.n} of ${hist.count}`, "small"),
    h("div", "cmp-row", pane(hist.first.id, "First check", hist.first.saved_at), pane(other.id, viewingFirst ? "Latest check" : "This check", other.at)),
    h("p", "lede", hist.compare_advice),
    actions("one", btn("Back to the result", closeOverlays, "btn compact")),
  );
  $("sheet").hidden = false;
}

// ---------- questions ----------
function stateForQuestions() { const o = shown(); return o ? o.verdict.state : ""; }
async function loadSuggestions() {
  const d = await getJSON("/questions?state=" + encodeURIComponent(stateForQuestions()));
  if (d) { S.suggestions = d.suggestions; if (S.screen === "questions") RENDER.questions(); }
}
async function ask(question) {
  const o = shown();
  const qa = { question, pending: true };
  S.thread.push(qa);
  if (S.thread.length > 6) S.thread.shift();
  RENDER.questions();
  const r = await post("/ask", { question, state: o ? o.verdict.state : "", sign_lines: o ? (o.sign_lines || []).map((l) => l.text) : [] });
  Object.assign(qa, r.ok && r.data ? r.data : NO_REPLY, { pending: false });
  if (r.ok && r.data) S.suggestions = r.data.suggestions;
  if (S.screen === "questions") RENDER.questions();
}

// ---------- staff readout ----------
function openStaff() {
  const o = shown();
  if (!o || !o.prediction) return;
  const p = o.prediction;
  const signs = (o.signs || []).map((s) => {
    const value = s.letter === "E" && currentHistory() && currentHistory().count > 1 ? `#${currentHistory().number}` : s.value === null ? "—" : s.letter === "D" ? (s.value * 100).toFixed(1) + "%" : s.letter === "C" ? String(s.value) : s.value.toFixed(2);
    const hist = currentHistory();
    const eHist = s.letter === "E" && hist && hist.count > 1;
    const pill = eHist ? `HISTORY · ${hist.count} SCANS` : s.tier === null ? (s.letter === "E" ? "NEEDS HISTORY" : "NO SCALE") : PILL[s.tier];
    const row = h("div", "srow", h("span", "l", s.letter), h("span", "n", SIGN_NAME[s.letter]), h("span", "v", value), h("span", "p", pill));
    row.lastChild.style.color = eHist ? HISTORY_ARC : ARC[s.tier];
    return row;
  });
  const fills = { benign: "#8C9BB0", pre_cancerous: "#C79A6B", malignant: "#E0645A" };
  const order = Object.keys(PROB_NAME).filter((k) => k in p.probs_pct);
  const probs = order.map((k) => [k, p.probs_pct[k]]).map(([k, v]) => {
    const bar = h("i"); bar.style.width = Math.max(1.5, v) + "%"; bar.style.background = fills[k] || "#8C9BB0";
    return h("div", "prob" + (k === p.label ? " flag" : ""), h("span", "n", PROB_NAME[k] || k), h("span", "t", bar), h("span", "v", v.toFixed(1) + "%"));
  });
  const stages = Object.entries(o.stage_ms || {}).map(([k, v]) => `${k} ${v}`).join(" · ");
  const gate = Object.entries(o.measured || {}).map(([k, v]) => `${k.replace(/_/g, " ")} ${Number(v).toFixed(2)}`).join(" · ");
  $("staff").replaceChildren(
    h("div", "eyebrow-row", eyebrow("CLINICAL READOUT · STAFF ONLY"), btn("Close ×", closeOverlays, "close")),
    signs.length ? h("div", "", signs) : null,
    h("div", "label", "MODEL CONFIDENCE (CALIBRATED)"),
    ...probs,
    h("div", "tech",
      `${o.model || "model"} · ${S.health && S.health.model && S.health.model.tta ? "TTA on" : "TTA off"} · ${p.flagged ? "flagged for a doctor" : "not flagged"} · top ${p.confidence_pct.toFixed(0)}%${o.forced ? " · photo check overridden" : ""}`,
      h("br"), `scan ${o.total_ms} ms (${stages})`,
      gate ? [h("br"), `gate: ${gate}`] : null),
  );
  $("staff").hidden = false;
}
function closeOverlays() { $("staff").hidden = true; $("sheet").hidden = true; }

// ---------- settings, keypad, erase, exit ----------
function renderKeypad(message) {
  S.code = "";
  const dots = h("div", "code-dots", message || "· · · ·");
  const keys = h("div", "keys", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "Clear", "0", "Cancel"].map((k) => btn(k, () => keypadPress(k, dots))));
  setScreen(eyebrow("SETTINGS · STAFF"), h1("Settings"), lede("Staff only. Enter the four-digit staff code."), h("div", "keypad", dots, keys));
}
async function keypadPress(k, dots) {
  if (k === "Cancel") return go("home");
  if (k === "Clear") S.code = "";
  else if (S.code.length < 4) S.code += k;
  dots.textContent = S.code ? "●".repeat(S.code.length) + " ·".repeat(4 - S.code.length) : "· · · ·";
  if (S.code.length < 4) return;
  const code = S.code;
  S.code = "";
  const r = await post("/staff", form({ code }));
  if (r.ok) { S.staffCode = code; go("settings"); } else renderKeypad("Wrong code");
}
async function changeSetting(key, on) {
  const r = await post("/settings", staffForm({ [key]: on ? "1" : "0" }));
  if (r.ok && r.data) S.settings = r.data.settings;
  else if (r.status === 403) S.staffCode = null;
  go("settings");
}
// Two taps, like "erase all": the first arms it for three seconds.
async function deleteViewed(e) {
  const b = e.currentTarget;
  if (b.dataset.armed !== "1") {
    b.dataset.armed = "1"; b.textContent = "Tap again to delete";
    setTimeout(() => { if (b.isConnected) { b.dataset.armed = ""; b.textContent = "Delete this scan"; } }, 3000);
    return;
  }
  const r = await post(`/saved/${S.viewing.id}/delete`);
  if (r.ok && r.data) S.stats = r.data.saved;
  S.viewing = null;
  await refreshLiveHistory();
  go("saved");
}

async function eraseAll(e) {
  const b = e.currentTarget, now = Date.now();
  if (now - S.eraseArmedAt > 3000) {
    S.eraseArmedAt = now;
    b.textContent = "Tap again to erase all";
    setTimeout(() => { if (b.isConnected) b.textContent = "End event — erase all"; }, 3000);
    return;
  }
  S.eraseArmedAt = 0;
  const r = await post("/erase", staffForm());
  if (r.ok) { S.spots = []; S.stats = r.data.saved; S.filter = null; S.viewing = null; forgetPhoto(); S.savedId = null; S.history = null; }
  go("settings");
}
async function quitKiosk(e) {
  const b = e.currentTarget;
  if (b.dataset.armed !== "1") {
    b.dataset.armed = "1"; b.textContent = "Tap again to exit";
    setTimeout(() => { if (b.isConnected) { b.dataset.armed = ""; b.textContent = "Exit kiosk"; } }, 3000);
    return;
  }
  const r = await post("/quit", staffForm());
  b.textContent = r.ok ? "Exiting…" : "Could not exit";
}

// ---------- boot ----------
async function boot() {
  tick(); setInterval(tick, 15000);
  const hl = await getJSON("/health");
  S.health = hl;
  if (hl) {
    S.settings = hl.settings || S.settings;
    S.stats = hl.saved || S.stats;
    S.camera = hl.camera ? "pi" : (navigator.mediaDevices && navigator.mediaDevices.getUserMedia ? "webcam" : "off");
  }
  loadSuggestions();
  loadSaved();
  go("home");
}
boot();
