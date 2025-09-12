# Logging Guide

This project records training runs and analysis with a persistent run_id to make correlation simple.

- logs/experiments_log.jsonl: JSON lines of run events (run_start, resume, run_end)
- logs/detailed_loss_analysis.json (or logs/loss_analysis.json): array of per-epoch analysis entries
- checkpoints: include run_id in metadata

## Finding the current run_id (PowerShell)

```powershell
# Show the last experiment log entry (contains run_id)
Get-Content logs/experiments_log.jsonl | Select-Object -Last 1
```

## Filtering by run_id

```powershell
# Replace <RUN_ID> with the value from experiments_log.jsonl

# All experiment-log entries for that run
Get-Content logs/experiments_log.jsonl | Select-String '"run_id":"<RUN_ID>"'

# All detailed loss analysis entries for that run
($a = Get-Content logs/detailed_loss_analysis.json | ConvertFrom-Json) | Where-Object { $_.run_id -eq "<RUN_ID>" }

# Summarize epoch, method, health for that run
($a = Get-Content logs/detailed_loss_analysis.json | ConvertFrom-Json) |
  Where-Object { $_.run_id -eq "<RUN_ID>" } |
  Select-Object epoch, method, health_score, timestamp |
  Format-Table -AutoSize
```

## Filtering by time window

```powershell
# Use the run_start/run_end timestamps from experiments_log.jsonl
# Replace the times below accordingly
($a = Get-Content logs/detailed_loss_analysis.json | ConvertFrom-Json) | Where-Object {
  ([datetime]$_.timestamp) -ge [datetime]"2025-09-12T10:05:00" -and
  ([datetime]$_.timestamp) -le [datetime]"2025-09-12T11:20:00"
}
```

## Notes
- run_id format: <config_name>_YYYYMMDD_HHMMSS
- Every run automatically gets a new run_id at startup (single source of truth is TrainingUtilities).
- Resumed runs log both a run_start (mode=Resume) and a resume record including checkpoint restore details and a shallow config diff.
