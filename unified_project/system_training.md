# System Training: Artistic Quality Reference

The model is "trained" via system prompting to produce high-quality artistic p5.js sketches.
All examples are **dynamic, animated, interactive** sketches. **The criterion is expressiveness and resonance.**

---

## Where It Lives in `app.py`

| Constant | ~Line | Purpose |
|----------|-------|---------|
| `FIRST_DATE_CODE` | 99 | Score 3-4 Processing source ("first date excitement") |
| `ARTISTIC_QUALITY_REFERENCE` | 121 | Rubric + score-7 examples → injected into system prompt on every code generation call |
| `SEED_REFERENCE_MESSAGE` | 990 | Score 3-4 examples (code + screenshots) → injected once at session start as a `HumanMessage` |
| `build_system_prompt()` | 299 | Assembles the final system prompt |

> Screenshots can't go in the system prompt — Gemini's `system_instruction` is text-only. Score 3-4 images are injected as multimodal `HumanMessage` content instead.

---

## Scoring Rubric (1-7)

```
1-2  Emotion named or symbolized only. Generic shapes, stock colors.
     Could be any emotion with a theme swap.

3-4  Emotion is in visual choices but not in rendering logic.
     You recognize it intellectually. "About" the emotion, not "made of" it.

5-6  Rendering logic has emotional texture — instability, tension, or weight in the math.
     But decorative elements still carry part of the load.

7    The rendering logic IS the emotion. You feel it before you understand it.
     Cannot re-theme without rewriting the core mechanic.
```

|       | EXPRESSIVENESS | RESONANCE |
|-------|----------------|-----------|
| **1-2** | Literal symbols. Imagery does all the work. | Recognize only. Re-labelable with zero changes. |
| **3-4** | Thematic abstraction. Core mechanic still generic. | Mostly intellectual. Re-labelable with minor tweaks. |
| **5-6** | Emotional texture in the math. | Feel before you understand. Hard but not impossible to re-label. |
| **7 ✦** | Technique IS the emotion. Could not predict it from the emotion word alone. | Cannot watch without feeling it. Re-labeling requires rewriting core code. |

---

## Score 3-4 Examples

Injected at session start via `SEED_REFERENCE_MESSAGE`.

**"first date excitement"** — 3D cube of smaller cubes rotating with `sin()`-driven pulsing. Pink/navy color scheme. Energetic visually, but the mechanic (rotating 3D geometry) is generic — could be "awe" or "dizziness" unchanged. Source: Processing/Java in `FIRST_DATE_CODE`.

**"head in the clouds"** — screenshot only. Drop at: `reference_images/score3_head_in_clouds.png`

**"sadness"** — screenshot only. Drop at: `reference_images/score3_sadness.png`

> Create `unified_project/reference_images/` and restart the server after adding images — the seed is built once at startup.

---

## Score 7 Examples

Included in `ARTISTIC_QUALITY_REFERENCE`, active on every code generation call.

### "feeling disheveled"
**Why 7:** Captures the viewer's webcam face and makes their OWN FACE look chaotic. Dark face regions become anchor points for random-walk "hair" strands. The viewer is the disheveled subject — the sketch borrows their identity to produce the feeling.
**Key techniques:** `createCapture(VIDEO)`, pixel darkness → dot radius, random-walk from dark regions.

```javascript
let capture; let isCaptured = false;
var NORTH=0,NORTHEAST=1,EAST=2,SOUTHEAST=3,SOUTH=4,SOUTHWEST=5,WEST=6,NORTHWEST=7;
var direction, posX, posY;
function setup() {
  createCanvas(640, 480);
  capture = createCapture(VIDEO, function() { isCaptured = true; });
  capture.size(640, 480); background(255);
}
function draw() {
  clear(); background(255);
  if (!isCaptured) return;
  capture.loadPixels();
  const stepSize = 10;
  for (let y = 0; y < height; y += stepSize) {
    for (let x = 0; x < width; x += stepSize) {
      const i = y * width + x;
      const darkness = (255 - capture.pixels[i * 4]) / 255;
      ellipse(x, y, stepSize * darkness, stepSize * darkness);
      if (darkness > 0.8) { posX = x; posY = y; drawHair(); }
    }
  }
}
function drawHair() {
  var diameter = 2, stepSize = 2;
  for (var i = 0; i <= 20; i++) {
    direction = int(random(0, 8));
    if (direction == NORTH) posY -= stepSize;
    else if (direction == NORTHEAST) { posX += stepSize; posY -= stepSize; }
    else if (direction == EAST) posX += stepSize;
    else if (direction == SOUTHEAST) { posX += stepSize; posY += stepSize; }
    else if (direction == SOUTH) posY += stepSize;
    else if (direction == SOUTHWEST) { posX -= stepSize; posY += stepSize; }
    else if (direction == WEST) posX -= stepSize;
    else if (direction == NORTHWEST) { posX -= stepSize; posY -= stepSize; }
    if (posX > width) posX = 0; if (posX < 0) posX = width;
    if (posY < 0) posY = height; if (posY > height) posY = 0;
    ellipse(posX + stepSize/2, posY + stepSize/2, diameter, diameter);
  }
}
```

### "feeling anxious"
**Why 7:** Anxiety = cognitive overload. Japanese literary characters rendered at sizes/colors driven by live audio FFT amplitude. Every frame: different size, different color, driven by sound. Illegibility + randomness + audio chaos = experienced overstimulation, not a symbol of it.
**Key techniques:** `p5.FFT waveform()`, per-character random color driven by `waveform[i]`, foreign script for visual density.

```javascript
let sound, fft;
let letters = '吾輩わがはいは猫である。名前はまだ無い。どこで生れたかとんと見当けんとうがつかぬ。'.split('');
function preload() { sound = loadSound('Catch_the_future.mp3'); }
function setup() {
  createCanvas(windowWidth, windowHeight);
  textFont('sans-serif'); textAlign(CENTER, CENTER);
  fft = new p5.FFT(); sound.amp(0.2); sound.loop();
}
function draw() {
  clear(); background(50);
  let waveform = fft.waveform();
  beginShape(); noFill(); stroke(255);
  for (let i = 0; i < waveform.length; i++) {
    vertex(map(i, 0, waveform.length, 0, width), map(waveform[i], -1, 1, height, 0));
  }
  endShape();
  for (let i = 0; i < letters.length; i++) {
    var j = Math.round(map(i, 0, letters.length, 0, waveform.length));
    fill(color(random(100,255), random(100,255), random(100,255)));
    textSize(Math.abs(waveform[j]) * 300 + 5);
    text(letters[i], map(j, 0, waveform.length, 0, width), map(waveform[j]*3, -1, 1, height, 0));
  }
}
```

---

## Adding New Examples

**Score-7 (system prompt):** Add a block to `ARTISTIC_QUALITY_REFERENCE` in `app.py` — title, WHY, KEY TECHNIQUES, CODE.

**Score 3-4 screenshot:** Drop PNG into `reference_images/`, add a `_load_ref_image(...)` call in `_build_seed_reference_message()`, restart server.
