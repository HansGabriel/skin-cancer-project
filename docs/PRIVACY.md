# Privacy

The kiosk photographs skin. This is what it does with the photo, and the paragraph the
paper's ethics section can cite.

## Nothing is stored

A photo lives in the server's memory only until the next photo replaces it, and in the
browser only until **Done**. There is no history, no saved cases, no database, no upload.
The log line per scan (`/tmp/dermascan_kiosk.log`) records the label, the confidence and
the timings — never the image, never a name.

The one exception is a switch a developer has to turn on by editing
`dermascan/config.py`: `SAVE_CAPTURES_DIR`. It keeps every capture as a JPEG beside its
photo-check numbers, so the gate can be calibrated against the real camera. It is off,
and it must stay off whenever the device is used with people rather than test images.

## Staff details

The **Details** button shows the model's three percentages and the scan timings. It
shows nothing about a person, so it is not locked. **Exit kiosk** asks for the staff code
when `~/.dermascan_passcode` exists, so a visitor cannot end the demo.

Pressing **Done** drops the photo from the page and from the server's memory.

## Before photographing students for the study

Capturing students' skin as research data is human-participants research under ISEF
rules (which Philippine NSTF fairs follow). Before the first photo: constituted IRB/SRC
approval, written parental permission plus minor assent, and the current forms (1, 1A,
1B, 4 + informed-consent statement). Demonstrations with photos shown on a phone screen,
or with the team's own skin, need none of that.
