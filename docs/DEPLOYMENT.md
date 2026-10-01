# Deployment

## Raspberry Pi 4 (the kiosk)

64-bit Raspberry Pi OS, Camera Module 2, the 7" 1024×600 HDMI touch panel.

```bash
sudo apt install -y python3-picamera2 surf git
cd ~/Documents
git clone <repo> skin-cancer-project && cd skin-cancer-project
git switch v2/simple-kiosk               # until it is merged to main
python3 -m venv --system-site-packages venv   # so apt's picamera2 is visible inside the venv
source venv/bin/activate
pip install -r requirements.txt
python -m kiosk.server                   # try it: http://127.0.0.1:8080 in any browser on the Pi
```

`EPIVUE.desktop` on the desktop runs `scripts/launch_kiosk.sh`, which starts the server,
waits for `/health`, opens `surf` full screen, and shuts everything down when **Exit kiosk**
(inside "Details", two taps) is pressed or the browser is closed. Copy the file to
`~/Desktop/` and mark it executable. The log is `/tmp/dermascan_kiosk.log`; one line per
scan looks like

```
scan status=ok label=benign conf=71 ms decode=18 gate=55 model=640 total=713
```

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

## Streamlit Community Cloud (public web demo)

Point the app at `cloud/streamlit_app.py` with `cloud/requirements.txt`; `runtime.txt`
pins Python 3.12. It is an upload box in front of the same `dermascan/` core. No camera,
no storage, no passcode needed — nothing is kept.

## Offline check before the fair

Wi-Fi off on the Pi, power-cycle, double-click the icon, run a full scan. Nothing in this
app contacts the network.
