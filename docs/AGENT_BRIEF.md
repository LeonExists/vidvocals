# Shared brief for all subagents (read PLAN.md first)

Repo: /home/user/vidvocals (branch claude/relaxed-faraday-syxjlq). Do NOT git commit or push; the coordinator does.
Only write the files your task owns. Other agents work in parallel in the same tree.

Environment facts (Linux build box, no GPU, 4 CPU cores, 15 GB RAM, ffmpeg 6.1 at /usr/bin/ffmpeg):
- Dev venv: /home/user/vidvocals/.venv (Python 3.12, torch 2.15 CPU-capable, torchaudio, transformers, speechbrain,
  librosa, soundfile, fastapi, uvicorn, pytest, httpx, scikit-learn, huggingface_hub).
  Activate: `source /home/user/vidvocals/.venv/bin/activate`. Add packages with
  `uv pip install --python /home/user/vidvocals/.venv/bin/python <pkg>` (PyPI works). Note new deps in your report.
- Always export: HF_HOME=/home/user/vidvocals/models/hf TORCH_HOME=/home/user/vidvocals/models/torch
  so downloaded models are shared between agents and not re-downloaded.
- Network: huggingface.co (models + datasets) and PyPI work. github.com only works for repos attached to this
  session (others return 403 "not enabled for this session"), dl.fbaipublicfiles.com returns AccessDenied,
  download-r2.pytorch.org is blocked. If you need something from a blocked host, look for a Hugging Face mirror
  first; otherwise say exactly which URL you need in your final report (the coordinator can attach GitHub repos).
- Resources are shared with 4 other agents: use torch.set_num_threads(2), keep RAM per process under ~4 GB,
  run heavy model jobs on short clips, and free models when done.
- Scratch files: /tmp/claude-0/-home-user-vidvocals/f3efe015-a2bc-59b3-accf-77853d5086ef/scratchpad/<your-task>/
- Target is Windows 11 + RTX 3080 10 GB + CUDA. Write code that works on both (pathlib, no shell-specific tricks).
- Final report: what you built, what you verified (commands + results), what you could not verify, open issues.
