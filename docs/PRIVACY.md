# Privacy

The kiosk photographs skin. This is what it does with the photo, and the paragraph the
paper's ethics section can cite.

## What is kept on the device

A photo lives in the server's memory only until the next photo replaces it, and in the
browser only until **Done**. There is no database and no upload; nothing leaves the device.

**Save this scan** keeps a 512 px copy of the photo, its result, a body site ("Left arm")
and the time, in a folder on the kiosk's SD card (`~/.dermascan/saved/`, two small files per
scan). No name, no account. Saved scans survive **Exit kiosk**, a crash, a reboot and a power
cut, so a group's history is still there the next morning (changed 11 October 2026 on the
mentor's decision; before that they lived in memory and vanished whenever the Pi was switched
off). They go when a visitor taps **Delete this scan** (no staff code needed), when staff tap
**End event — erase all**, or, past 60 saved scans, when the oldest is dropped.
The log line per scan (`/tmp/dermascan_kiosk.log`) records the label, the confidence and
the timings — never the image, never a name.

**The SD card therefore carries photos until End event.** Staff should tap **End event —
erase all** before the card or the Pi leaves their hands, and should tell visitors before
they save. Nothing on the card names a person.

The one exception is a switch a developer has to turn on by editing
`dermascan/config.py`: `SAVE_CAPTURES_DIR`. It keeps every capture as a JPEG beside its
photo-check numbers, so the gate can be calibrated against the real camera. It is off,
and it must stay off whenever the device is used with people rather than test images.

**Saved scans are not locked.** Anyone at the kiosk can open the Saved tab, see the
photos saved during the event, and delete one. Nothing there names a person (only a body
site such as "Left arm"), and deleting is allowed so visitors can remove their own. Staff
should tell visitors this before they save, and tap **End event — erase all** at the end.
Past 60 saved scans the oldest is dropped, so a long-running group's "first check" can
become its oldest remaining one.

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
