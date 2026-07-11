# Integrating VoiceForge into other apps

VoiceForge is a plain HTTP service on `http://127.0.0.1:8484`. Anything that
can `fetch` can speak. Start it with `VoiceForge.bat` (or programmatically:
`envs\server\Scripts\python.exe -m uvicorn app:app --port 8484 --app-dir server`).

## Mythos gallery — Chat Lab speaks

Backend proxy (recommended so the frontend needs no new origin), e.g. in
`backend/main.py`:

```python
@app.post("/api/chat/speak")
def chat_speak(body: dict):
    import urllib.request, json as _json
    req = urllib.request.Request(
        "http://127.0.0.1:8484/api/tts",
        data=_json.dumps({"text": body["text"], "voice": body.get("voice")}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return Response(content=r.read(), media_type="audio/wav")
```

Frontend (ChatLab message → speaker button):

```ts
const r = await fetch("/api/chat/speak", { method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: message.text }) });
new Audio(URL.createObjectURL(await r.blob())).play();
```

VRAM note: the gallery's VLM/LLM and VoiceForge share the 8 GB GPU. VoiceForge
auto-unloads after 15 idle minutes (Settings), and you can force it with
`POST /api/engines/chatterbox/unload`.

## OpenAI-compatible clients

Any tool that supports a custom OpenAI TTS base URL works:

```
base_url = http://127.0.0.1:8484/v1
POST /v1/audio/speech  {"model": "voiceforge", "voice": "<voice-slug>", "input": "..."}
```

(`voice` may be omitted → the default voice from Settings; only `wav` output.)

## Claude Code / scripts / future assistant

```
C:\Users\AdministratorCorrupt\Desktop\Claude Projects\VoiceForge\say.cmd -NoWait "text here"
```

or `POST /api/speak {"text": "...", "wait": false}` — generates and plays on
the PC's speakers server-side (no client audio stack needed). A `speak` skill
is installed at `~/.claude/skills/voiceforge-speak/` so Claude Code sessions
can announce things aloud.
