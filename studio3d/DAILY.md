# Why It Moved — daily 3D clip runbook (owner Tri, Nov 4 2026)

Direction: chase the latest news. Big moves in big companies that are trending: biggest droppers (>= 6-7%), companies backed by or tied to big tech, big-tech-wide stories. Hook -> body -> reward. No advice, no predictions, no real logos/faces, no "cartoon"/"not advice" wording. Kokoro am_michael, >= 30 s, -20 LUFS, free tools by default.

## Setup (fresh container)
1. `git clone https://github.com/ductringuyen-0618/whyitmoved-media /tmp/media` (add_repo push access first).
2. `mkdir -p /home/claude && tar xzf /tmp/media/studio3d/studio2-code.tgz -C /home/claude` (paths are absolute /home/claude/studio2/...). Read /home/claude/studio2/STUDIO.md and eng3d/README.md.
3. `cd /home/claude/studio2/eng3d && npm install` then `npx playwright install chromium` only if chromium is missing (it is normally preinstalled). `pip install kokoro-onnx soundfile numpy pillow --break-system-packages`. Kokoro model + voices download on first use into ~/.cache/stockshort/.
4. Reference clip to copy: /home/claude/studio2/twot (lines.json, build/build_shots.py, build/overlay.py, build/final.py, audio/stems.py, vo_measure.py, exact_timing.py). Copy to /home/claude/studio2/<slug>/ and adapt. exact_timing.py needs a patched duration model (/tmp/claude-0/kokoro_dur.onnx) that is NOT in the repo: if missing, use vo_measure.py word timing (vo_timing.json) and write it as vo_exact.json in the same format.

## Pipeline (see twot/ for a worked example)
facts pack (sourced, with a "does not claim" list) -> 12-14 short lines (<= ~110 words, >= 30 s) -> VO first -> word timing -> shots.json (one shot per line, characters named by company, props: coins, scale, server, signs, arrows) -> `node render.mjs --shots ../<slug>/shots.json --out ../<slug>/frames --from 0 --to N --fmt jpeg --skip 1` (about 4.4 s/frame at 1080x1920 on 2 cores; 50 s = 1500 frames = ~110 min: run in background and poll) -> overlay.py chips/title/attribution/CTA -> stems.py + pilot2/audio/mix.py to -20 LUFS -> final.py. Characters in chars.js: INTEL, NVIDIA, AMD, ALPHABET, AMAZON (add new ones by copying a builder: original designs only).
To cut render time, a 30-35 s clip (~1000 frames) is preferred.

## Gates
Independent fact-check agent on every number (wording like "on track to", "more than", "so far" when sources conflict), independent motion/framing check on a contact sheet, ffprobe length, LUFS -20.

## Publish
Copy mp4 to /tmp/media/<YYYY-MM-DD>/<slug>.mp4, git add/commit/pull --rebase/push, then Metricool createScheduledPost (brand 7839040, America/Los_Angeles, youtube short + tiktok + facebook REEL, raw.githubusercontent.com media URL, isAiGeneratedContent/isAigc true). Add a LEDGER.csv row (kind=3d). Report to Tri with SendUserMessage + SendUserFile.
