# Remix

A Dockerized toolkit that takes a video apart so it can be remixed: speech stem, background stem, word-timed script, shots, keyframes and contact sheets. It then builds remixes from those parts: new music, new AI voices, captions, and a full restyle of the picture.

It is built to be driven by an AI coding agent (it was written with Claude Code). The agent runs the `remix` command in the container, reads the outputs, writes the creative parts (the blueprint of a video, a rewritten script, the look of a restyle) and checks the results. No model API key lives in the container; the paid media models are Google Cloud's, called with a short-lived access token passed in from the host.

## What it does

| Stage | Command | What you get |
|---|---|---|
| Take apart | `remix decompose <video>` | `audio/speech.wav` and `background.wav` (Demucs), a word-timed transcript (faster-whisper), shot boundaries, keyframes and contact sheets (PySceneDetect), `overview.md` |
| Blueprint | `remix blueprint init` / `check` | Themes, tone, structure, a description of every shot, measured pacing and tempo. Written by the agent, validated by the tool |
| Cost | `remix cost estimate` / `report` | Every remix route priced for this video; a ledger of every paid call; credit left |
| New music | `remix music`, `remix audio-remix` | A generated track (Lyria), mixed under the picture at the level the old background had |
| New voices | `remix lines`, `remix voices` | The script as timed lines, spoken by stock AI voices (Gemini TTS) with vocal cues, each line fitted to the time of its original utterance |
| Captions | `remix captions` | `.srt` and styled `.ass` from the same lines, one color per speaker |
| New picture | `remix restyle` | Every shot redrawn by a video model (Gemini Omni) from a `look.json`, pilot first, resumable, captions burned in |
| Make from a prompt | `remix make new / plan / shots / voice / music / render / check` | A whole video from a brief: Gemini drafts the plan, Veo generates the shots (pilot tier first, then final), a stock voice and a score are added, every word on screen is drawn in code, and the result is checked. Also in the web app |
| Audio provider | `--provider gemini\|elevenlabs` on `remix make voice / music`, `remix voices`; a picker in the web app | Voice and music from Gemini and Lyria (automatic, paid) or from your own ElevenLabs account: through the official SDK when `ELEVENLABS_API_KEY` is set, otherwise from takes saved into the project's `eleven/` folders. Either way every take is fitted to its slot and checked the same way |
| Check | `remix verify` | Runs by itself after every remix: script against audio, captions and picture, and voice against picture timing |

There is also a local web app (`start-web.cmd`, then http://localhost:8765) that walks one video through these stages as gates, with the price on every paid button and an explicit approval at the end of each.

## Running it

Requirements: Docker. For the paid steps, the Google Cloud CLI signed in to a project with the Agent Platform API enabled. An NVIDIA GPU is optional.

```bash
docker compose build
docker compose run --rm remix decompose inbox/<file>
docker compose run --rm remix list
docker compose run --rm --entrypoint pytest remix -q /app/tests
```

`./src` and `./tests` are bind-mounted, so code changes apply without a rebuild. Videos go in `inbox/`; everything made from them lands in `projects/<slug>/`. Neither folder is part of this repository.

Paid steps take a one-hour token from the host:

```bash
GOOGLE_ACCESS_TOKEN="$(gcloud auth print-access-token)" GOOGLE_CLOUD_PROJECT=<your-project> \
  docker compose run --rm -e GOOGLE_ACCESS_TOKEN -e GOOGLE_CLOUD_PROJECT remix voices <slug> --name <remix> --pilot 4
```

GPU image for the stems and transcript steps:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml build
docker compose -f compose.yaml -f compose.gpu.yaml run --rm remix decompose inbox/<file>
```

## Rules the tool enforces

- **Cost.** Every paid call is estimated first and recorded in a ledger. Anything over the cap (`CAP_USD`, 5 dollars by default) needs an explicit yes. Visual runs start with a pilot of a few shots, and piloted shots are never paid for twice.
- **Rights.** Each project is marked as your own video, a client's, or someone else's used only as a template. Modes that reuse footage refuse template projects.
- **One script.** `lines.json` is the single source for voices, captions and the picture. Each part records the script it was made from; a mismatch is refused or regenerated.
- **Sync.** Line slots are anchored to where the original speech really is (measured on the speech stem, not taken from transcript stamps). A line's first word is placed on that onset and its tempo changed by at most 12 percent to end with it; lines that do not fit are spoken again at another pace. The check measures every line against the original utterance and fails a video whose voices start or stop away from where the mouths move. Each shot's prompt names who speaks, in which seconds, and who keeps their mouth closed.
- **Checked before handover.** Every command that makes a remix video runs the check and exits with a distinct code when it fails. For a regenerated picture the check writes frame sheets that must be read before it can pass.
- **Voices.** Stock voices only. No voice cloning and no lip-sync models.

## Layout

```
src/remix/
  cli.py          the remix command
  steps/          audio, stems, transcript, shots
  blueprint.py    blueprint skeleton, validation, measured metrics
  cost.py         price table, estimates, ledger, credit
  prices.yaml     dated provider prices
  google.py       calls to Lyria, Gemini TTS and Gemini Omni
  remixes.py      the audio mixer
  voices.py       lines, vocal cues, anchoring, fitting, captions
  restyle.py      per-shot prompts, pilots, full restyle
  verify.py       the script and sync check, frame sheets
  web.py          the local web app (standard library only)
  webapp.html     its single page
tests/            pytest suite; runs without any model or paid call
```

## Status

Built: decompose, blueprint, cost tracking, music, voices, captions, restyle, the check, the web app. Not built yet: regeneration from the blueprint without the original footage, a motion-graphics rebuild as code, a translate flow, splitting shots longer than 15 seconds, and a provider price watch.

Provider model names, endpoints and prices in this code were read from Google's documentation in November 2026 and will drift; `prices.yaml` carries the dates.
