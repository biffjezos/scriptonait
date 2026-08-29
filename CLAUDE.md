# Standing rules for this project

These are explicit, repeated instructions from the project owner. Follow them
without being asked again.

- **Never add explanatory or decorative UI text.** No hints, no taglines, no
  "here's what this does" prose in the app. Field labels and button text only
  — functional, not explanatory.
- **Never hardcode a setting that already has a home in the UI.** If a value
  (temperature, max tokens, device choice, prompt, etc.) is already exposed
  as a control somewhere, every feature that needs it reads from that same
  control. Never a second hidden copy, never a literal fallback value baked
  into the code.
- **File saves use the browser's real save picker** (`showSaveFilePicker`)
  so the user can choose the name and location, not a silent
  hardcoded-filename download — for every export/save action, when the API
  is available.
- **Settings belong in the Settings tab.** Keep other tabs (Training,
  Inference) focused on the actions themselves and live status, not
  configuration.
- **CPU-preferred operations must not block GPU work.** When something is
  set to run on CPU (e.g. inference during training), it must run
  concurrently, not pause whatever's running on the GPU.
- **Autosave-to-file saves the whole project** (checkpoint + corpus +
  history + settings), not just model weights — a single file left behind
  after a crash has to be enough on its own.
