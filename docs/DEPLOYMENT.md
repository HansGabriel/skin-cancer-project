# Deployment

## Raspberry Pi 4 (the kiosk)

64-bit Raspberry Pi OS, Camera Module 2, the 7" 1024×600 HDMI touch panel.

```bash
sudo apt install -y python3-picamera2 surf git
cd ~/Documents
git clone <repo> skin-cancer-project && cd skin-cancer-project
git branch --show-current                # should say main
python3 -m venv --system-site-packages venv   # so apt's picamera2 is visible inside the venv
source venv/bin/activate
pip install -r requirements.txt
python -m kiosk.server                   # try it: http://127.0.0.1:8080 in any browser on the Pi
```

`EPIVUE.desktop` on the desktop runs `scripts/launch_kiosk.sh`, which starts the server,
waits for `/health`, opens `surf` full screen, and shuts everything down when **Exit kiosk**
(inside "Details") is pressed or the browser is closed. Exit asks for the staff code on an
on-screen keypad when one is set; otherwise it is two taps. Set the code once:

```bash
printf %s '2468' > ~/.dermascan_passcode && chmod 600 ~/.dermascan_passcode   # pick your own 4 digits
```

Copy `EPIVUE.desktop` to `~/Desktop/` and mark it executable. The log is `/tmp/dermascan_kiosk.log`; a visitor's
check and scan look like

```
check status=pass focus=41.2 brightness=128 ms decode=18 gate=55 total=73
scan status=ok label=benign conf=71 ms decode=0 gate=0 model=640 signs=9 total=649 check=reused
```

`check=reused` means the scan used the photo check's decode and gate instead of running
them again. `python scripts/bench_scan.py` prints the same stages as medians over ten rounds;
paste its output into `docs/METRICS.md` when the Pi's numbers are measured.

**Autostart at boot** (optional): add to `~/.config/labwc/autostart`

```
wlr-randr --output HDMI-A-1 --mode 1024x600 --scale 1 &
~/Documents/skin-cancer-project/scripts/launch_kiosk.sh &
```

The first line matters: the panel's EDID advertises 1920×1080 and the compositor will
pick it, which makes everything microscopic.

**Updating the Pi:** `cd ~/Documents/skin-cancer-project && git branch --show-current`
(check it first — a pull on the wrong branch has bitten before), then `git pull`.

**Pulling the camera out of a jam:** the camera is opened once when the server starts
and never reopened. If the live view is dead, restart the kiosk; if it is dead at boot,
`libcamera-hello` in a terminal tells you whether the camera itself is seen.

## Laptop (development)

`scripts/run_dev.sh` runs the same server with auto-reload. No camera module: the page
uses the browser's camera (works on `localhost`) or a file picker.

## Public web demo

The same `dermascan/` core behind an upload box (`cloud/streamlit_app.py`). No camera, no
storage, no passcode: nothing is kept between uploads. Accuracy is the kiosk's, see
`docs/METRICS.md`; the page says so.

### Railway (primary)

The repo carries everything Railway needs: `Dockerfile` (python 3.12-slim, installs
`cloud/requirements.txt`, copies `dermascan/ models/ cloud/ .streamlit/`, binds Streamlit to
`$PORT`), `.dockerignore` (keeps the venv, training, kiosk and tests out of the image) and
`railway.json` (Dockerfile builder, health check on `/_stcore/health`, restart on failure).
No environment variables, no secrets.

Once, in the Railway dashboard:

1. **New Project → Deploy from GitHub repo**, pick this repo, branch `main`, root directory `/`.
   Railway reads `railway.json` and builds the Dockerfile; the first build takes a few minutes.
2. **Settings → Networking → Generate Domain.** Leave the port on `8501` if it asks.
3. Open `https://<your-domain>/_stcore/health` — it must say `ok` — then the page itself:
   upload a photo, open **Details for staff**, check the timings.
4. Every push to `main` redeploys. Watch the **Deployments** tab after a model swap.

Cost: free during the trial, then the Hobby plan (about USD 5 a month including usage).
The container idles at a few hundred MB of RAM; one replica is plenty for a demo.

Locally, the same image:

```bash
docker build -t epivue-demo . && docker run --rm -p 8501:8501 epivue-demo
curl -s localhost:8501/_stcore/health     # ok
```

### Streamlit Community Cloud (fallback)

Still works: point the app at `cloud/streamlit_app.py` with `cloud/requirements.txt`;
`runtime.txt` pins Python 3.12. Keep it until the Railway URL has been checked, then
remove this subsection and the Streamlit Cloud mentions in `README.md` and `AGENTS.md`.

## Offline check before the fair

Wi-Fi off on the Pi, power-cycle, double-click the icon, run a full scan. Nothing in this
app contacts the network.
