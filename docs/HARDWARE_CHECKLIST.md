# Hardware checklist — before every demo

Written for the student team. Do these in order; most "the kiosk is broken" reports
are one of the first three.

## The message that started this list

> *"Good evening sir, our research adviser checked it earlier but it acted up again — the yellow
> paper-like thing behind the monitor has to be pressed for the display to show, so our adviser
> said if possible we should mount it in the 3D-printed case first because it is fragile without it."*

The yellow strip is the **flat ribbon cable (FPC)** between the panel's driver board and the
LCD glass. If pressing it makes the picture appear, it is not seated in its connector. That
is a hardware fault; no software change will fix it.

## 1. Reseat the display ribbon

1. Power everything off. Unplug the Pi.
2. On the driver board, find the connector where the yellow ribbon enters. It has a small
   latch — a dark bar that flips up (or slides out ~1 mm) on the ribbon side.
3. Open the latch gently with a fingernail. Pull the ribbon straight out.
4. Check the gold contacts are clean and the ribbon is not creased. Never fold it.
5. Push the ribbon straight back in, all the way, **square to the connector**, then close
   the latch. It should not pull out with a light tug.
6. Lay the ribbon flat and hold it down with a small piece of tape so it cannot flex.

If the picture still needs pressure, the connector or ribbon is damaged — ask for a
replacement panel rather than demoing with it.

## 2. Mount in the 3D-printed case

The adviser is right: do this before any more handling. The ribbon fails from being flexed
every time the panel is picked up. Once in the case, nothing should touch the back of the
panel again.

## 3. Power and cables

* The Pi runs from the **official 5 V 3 A power supply** (the one with the USB-C plug).
  A phone charger browns out the Pi mid-scan and reboots it.
* The panel's **one micro-USB cable goes to a Pi USB port** — it carries both power and
  touch. Do not plug it into a wall charger (touch stops working) and do not add a second
  cable.
* HDMI from the Pi to the panel. "HDMI connected" in software does **not** prove the panel
  has power: the panel's little chip is powered through the HDMI cable even when the panel
  itself is dead. If the screen is black, check the micro-USB first.
* Camera ribbon: same rules as the display ribbon — straight, flat, latched, blue side as
  marked on the Pi's camera connector.

## 4. First boot check (2 minutes)

1. Desktop appears on the panel at the right size (text readable, nothing cut off). If
   everything is microscopic, the panel came up at 1920×1080: in a terminal,
   `wlr-randr --output HDMI-A-1 --mode 1024x600`.
2. Touch works: tap the E.P.I.V.U.E. icon.
3. The kiosk shows **LIVE** in the camera circle within ~10 seconds.
4. Point it at a mole or a photo of one, take the photo, check the spot. A result should
   appear in **under 5 seconds**. The "Details" button shows the timing.
5. Tap **Done**, then **Start a skin check** again: the live picture must come back at once.
   Do this three times. If it ever freezes, note what you did and tell the mentor; the
   log is in `/tmp/dermascan_kiosk.log`.

## 5. If the picture is blurry up close

The Camera Module 2 is focused for about 1 metre from the factory. A mole 5 cm away will
be soft and the kiosk will say "too blurry". The lens can be turned (gently, with the
small plastic tool or fine tweezers) to refocus closer. Do this once, with the mentor,
then do not touch it again. The macro lens with the white diffuser ring is the better fix
when it is fitted.
