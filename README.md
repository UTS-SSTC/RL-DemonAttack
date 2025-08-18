# RL-DemonAttack

## How to initialize?

1. Install [Scoop](https://scoop.sh/) \[Powershell\]</br>

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
   ```

2. Install [uv](https://github.com/astral-sh/uv) \[Cmd\]</br>

   ```cmd
   scoop install main/uv
   ```

3. Create virtual environment and sync dependencies \[Cmd\]</br>

   ```cmd
   uv venv && uv sync
   ```

