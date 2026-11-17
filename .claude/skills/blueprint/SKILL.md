---
name: blueprint
description: Write the blueprint for a decomposed Remix video - themes, tone, structure, visual and audio style, and a description of every shot. Use after `remix decompose` when asked to analyze a video, extract its themes or style, or before planning any remix.
---

# Writing a blueprint

The blueprint is the structured description every remix starts from. Code measures what can be measured; you supply what needs eyes and judgment.

## Steps

1. `docker compose run --rm remix blueprint init <slug>` writes `projects/<slug>/blueprint.json` with shot timings, the words spoken in each shot, and measured metrics. Everything else is empty.
2. Read `overview.md`, then **every** contact sheet in `shots/sheets/` (tiles are labeled `S<shot> <time>`). Open a keyframe from `shots/keyframes/` at full size whenever a tile is too small to read text or judge detail.
3. Fill in the empty fields of `blueprint.json`. Each shot needs only `index` plus the fields you write: `check` restores `start`, `end`, `spoken` and `metrics` from the measurements, so rewriting the whole file is safe.
4. `docker compose run --rm remix blueprint check <slug>` lists anything unfinished. Fix it and rerun until it reports the blueprint complete; it then renders `blueprint.md`.

## What goes in each field

| Field | What to write |
|---|---|
| `summary` | Two or three sentences: what the video is and what it does to the viewer. |
| `format` | The genre and build, e.g. "vertical short, narrated stock-footage montage with burned-in captions". |
| `audience` | Who it is made for, judged from content and style. |
| `themes` | 3-6 ideas the video is about. |
| `tone` | 2-5 words for how it feels. |
| `structure` | 3-7 beats covering the runtime in order: `{"beat", "start", "end", "what"}`. Name beats by function (hook, setup, escalation, payoff, cta). |
| `visual_style.look` | One paragraph a stranger could use to recognize the style. |
| `visual_style.palette` | 3-6 dominant colors as plain names or hex. |
| `visual_style.lighting`, `.camera`, `.editing` | How it is lit, framed and moved, and how it is cut (pace, transitions). |
| `visual_style.on_screen_text` | Caption and title style: font feel, color, position, animation. "none" if there is none. |
| `audio_style.voice` | Who speaks and how the script is delivered, from the transcript only (pace from the metrics, register from the wording). |
| `audio_style.music`, `.sound_design` | Measured facts only, plus anything the picture or script makes certain. See the rule below. |
| `characters` | Recurring people or figures: `{"id", "description", "shots"}`. Empty list if none recur. |
| `shots[].description` | One or two sentences: subject, action, setting. |
| `shots[].camera` | Framing, angle and movement. Movement can only be judged when a shot has more than one keyframe. |
| `shots[].on_screen_text` | Text that is part of the picture (titles, labels, signs, screens), verbatim, as a list. Empty list if none. Burned-in captions that repeat the speech are described once in `visual_style.on_screen_text`, not per shot. |
| `shots[].speaker` | The `id` of the character who does most of the talking in the shot, or `""` when nobody speaks. The transcript has no speaker labels, so judge from who is on screen and from caption styling. |
| `shots[].role` | One of: hook, title, setup, point, proof, demo, b-roll, transition, payoff, cta, outro, other. |
| `remix_notes.essential` | What makes the video work and must survive any remix. |
| `remix_notes.swappable` | What a remix could change freely: subject, setting, style, music, voice. |
| `remix_notes.risks` | What could block or weaken a remix: recognizable people, logos, baked-in text, lip-synced speech, shots longer than 15 s, copyrighted music. |

## Rules

- **You cannot hear.** Never describe how music or a voice sounds. For music, state the measured tempo and how much of the runtime has background audio, and say plainly that it has not been listened to.
- **Describe what the frames show.** One keyframe is one instant; do not invent action between frames. If a shot's content is unclear, say so in the description.
- **People:** describe clothing, framing and action. Do not name a real person unless the video itself names them in speech or on screen, and do not guess pronouns; use "they".
- **Third-party videos** are templates. The blueprint describes structure and style; it does not make the footage, music or voices reusable.
- Quote on-screen text exactly as written, including casing.
- **Keep numbers out of the prose.** Word counts, pace, cut rate and tempo are rendered under "Measured" and change when a video is re-processed; refer to them ("brisk pace, see Measured") instead of copying them.
- If burned-in captions disagree with the transcript, say so under `remix_notes.risks`: the captions are usually right.
