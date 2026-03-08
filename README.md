# Executive Inbox OpenEnv Submission

This repository contains our Executive Inbox environment submission for the OpenEnv hackathon.

## Contents
- `envs/executive_inbox/`: the environment package
- `scripts/executive_inbox_unsloth_reinforce.py`: minimal Unsloth training script
- `scripts/verify_executive_inbox_space.py`: remote HF Space verification script
- `scripts/executive_inbox_submission_runs.ipynb`: judge-facing Jupyter notebook with benchmark cells

## Deployment
OpenEnv stable release `0.2.1` deployed on Hugging Face Spaces:
- https://huggingface.co/spaces/hoony/executive-inbox

## Best benchmark result
- Train: average reward `1.151`, solve rate `17/20 = 0.85`
- Held-out base eval: average reward `1.0405`, solve rate `15/20 = 0.75`
- Held-out trained eval: average reward `1.2765`, solve rate `19/20 = 0.95`

## Remote verification
We verified:
- remote `reset`, `step`, and `state`
- a minimal Unsloth training run against the deployed HF Space
