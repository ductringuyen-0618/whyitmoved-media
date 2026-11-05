# Cartoon-audio-short runbook (daily hot-topic clip) — preferred style from Nov 5 2026

Tri prefers this style over the old 3D-studio look. Use it for the daily hot-topic clip. Same rules as DAILY.md: >= 30 s, hook -> body -> reward, no advice/predictions, no real logos or faces, no "cartoon"/"not advice" wording, sourced numbers only, free tools.
Worked examples inside the bundle: examples/brannan (42 s, story.js/main.js/audio.py) and examples/franklin (34 s, full web/ project).

## Setup (fresh container)
1. Clone the repo to /tmp/media (see DAILY.md step 1).
2. `mkdir -p /home/claude && tar xzf /tmp/media/studio3d/cartoon-kit.tgz -C /home/claude` -> /home/claude/cartoon-kit
3. `bash /home/claude/cartoon-kit/tools/setup.sh /home/claude/shorts/<slug>` (needs node >= 22.6, ffmpeg, Playwright Chromium; fetches three.js r160 via npm). Then `cd /home/claude/shorts/<slug>`.
4. `pip install kokoro-onnx soundfile numpy pillow --break-system-packages`.

## Voice (no recording available in scheduled runs)
Kokoro am_michael. Generate each script sentence separately with the loader in /tmp/media/tools/stockshort.py (`import sys; sys.path.insert(0,'/tmp/media/tools'); import stockshort as s; s.synth(text, wav, 'am_michael', 1.0)`), join them with 0.40 s of digital silence between sentences (0.7 s before the reward line), write voice.mp3 (ffmpeg) and script.txt (one sentence per line). The clean silences let `python3 timing.py voice.mp3 script.txt --out work/` align phrases correctly (silence-align mode; no whisper installed). Read work/phrases.json: every phrase must sit in its own speech segment. If Tri has uploaded an ElevenLabs recording, use that instead (the sandbox cannot download from elevenlabsusercontent.com).

## Build
1. Facts pack with a "does not claim" list; script 90-115 words (~30-40 s); company names appear as plain text signs/labels only.
2. Copy examples/franklin/web/{main.js,story.js} over the project's web/ as a starting point (world = town street, signs with dynamic text via mkSign/setSign, coin piles, chest, clock; cast in roman.js OUTFITS, then `cat web/toonbase.js web/roman.js > web/toon.js`). Rewrite worldAt/actorsAt/fxAt per phrase from work/phrases.json frames (one shot per phrase or phrase pair; SHOTS contiguous, TOTAL = frame count). For market stories reuse signs as price boards ("-9%" red, "+12%" green), coin piles shrinking/growing, and company-named buildings (text only).
3. Framing rule (portrait): horizontal half-width = d * tan(fov/2) * 0.5625. A wide shot needs d ~ 14-18 at fov 38-42. Keep subjects within +-2 m of center for close shots. Check stills first: `node render.mjs stills/a "f1,f2,..." 1 0.9` then `python3 sheet.py stills/a stills/sheet_a.jpg` and Read it. Fix cut-off text, overlaps, empty frames.
4. Full render in the background: `nohup node render.mjs frames "0-N" 1 0.95 > render.log 2>&1 &` (~2.8 s/frame, 2 cores; ~50 min for 1000 frames). Poll with an until-loop on `ls frames | wc -l`.
5. Audio: adapt examples/franklin/audio.py (music bed + SFX cue table by frame: plinks, whooshes, ticks, thunks, ching), run `python3 audio.py && python3 mix.py voice.mp3 music.wav sfx.wav` (music/SFX at 50 % of the voice's gated RMS).
6. `bash /home/claude/cartoon-kit/tools/finish.sh "<Title>"` -> out/<Title>.mp4 plus freeze/black QA and out/qa_sheet.jpg (Read it). Expect no freeze/black lines.

## Gates and publish
Independent fact-check agent on every number; contact-sheet check; ffprobe length >= 30 s. Then publish exactly as in DAILY.md (copy mp4 to /tmp/media/<YYYY-MM-DD>/<slug>.mp4, pull --rebase, push, Metricool createScheduledPost, LEDGER row kind=cartoon, SendUserMessage + SendUserFile).

## Modern finance world + action kit
Newer: see ACTIONS.md inside cartoon-kit.tgz and examples/finance-modern (modern city, bank, chip tower, USD stacks, vest crowd) plus examples/action-reel. Use these for hot-topic market shorts instead of the colonial town. New: cry/fall/faint/jump/car/rocket/moon.
