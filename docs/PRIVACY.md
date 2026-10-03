# Privacy

The kiosk photographs skin. This is what it does with the photo, and the paragraph the
paper's ethics section can cite.

## Nothing is written to disk

A photo lives in the server's memory only until the next photo replaces it, and in the
browser only until **Done**. There is no database and no upload.

**Save this scan** keeps a 512 px copy of the photo, its result and a body site ("Left
arm") in the server's memory, so the visitor can reopen it from **Saved** during the
event. No name, no account. It is never written to the SD card, and it is gone when
staff tap **End event — erase all**, on **Exit kiosk**, on a crash, or when the Pi is
switched off. At most 60 scans are held; past that the oldest is dropped.
The log line per scan (`/tmp/dermascan_kiosk.log`) records the label, the confidence and
the timings — never the image, never a name.

The one exception is a switch a developer has to turn on by editing
`dermascan/config.py`: `SAVE_CAPTURES_DIR`. It keeps every capture as a JPEG beside its
photo-check numbers, so the gate can be calibrated against the real camera. It is off,
and it must stay off whenever the device is used with people rather than test images.

## Staff details

**Details for staff** shows the model's three percentages, the A B C D E numbers and the
scan timings. It shows nothing about a person, so it is not locked (staff can hide the
button in Settings). **Settings** — and with it **End event — erase all** and **Exit
kiosk** — asks for the staff code when `~/.dermascan_passcode` exists, so a visitor
cannot change the kiosk, erase other people's saved scans, or end the demo.

## Questions

The Questions tab answers from `dermascan/answers.json`, a fixed set of written answers.
Questions are matched on the device; nothing is sent anywhere and no language model
runs. The answers were written by the project team; each shows "not yet reviewed by a
doctor" until a health professional fills in `reviewed_by` and `reviewed_date`.

Pressing **Done** drops the photo from the page and from the server's memory.

## Before photographing students for the study

Capturing students' skin as research data is human-participants research under ISEF
rules (which Philippine NSTF fairs follow). Before the first photo: constituted IRB/SRC
approval, written parental permission plus minor assent, and the current forms (1, 1A,
1B, 4 + informed-consent statement). Demonstrations with photos shown on a phone screen,
or with the team's own skin, need none of that.
