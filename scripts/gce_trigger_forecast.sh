#!/bin/bash
# Trigger the "Predict & Deploy Forecasts" workflow from the GCE host's cron.
#
# GitHub's own `schedule:` trigger starts this workflow 5-7 h late, so the VM fires it at
# fixed times through `workflow_dispatch`, which starts within seconds. The workflow's own
# cron is kept only as a fallback if the VM is down (see the workflow file).
#
# Times (UTC) are set from when ECMWF IFS runs reach Open-Meteo (~7.3 h after their start,
# measured with scripts/measure_openmeteo_delay.py):
#   04:30  morning forecast, on the previous day's 18z run (available ~01:15)
#   09:00  update, on today's 00z run (available ~08:00-08:30)
# Forecast files are named by run date and a later run overwrites the earlier one.
#
# Setup on the VM (once):
#   1. Create a fine-grained token at https://github.com/settings/personal-access-tokens:
#      repository access = AmedeeRoy/defile-migration-forecast only,
#      permissions = Actions: Read and write (nothing else).
#   2. mkdir -p ~/.config/defile && install -m 600 /dev/null ~/.config/defile/github_token
#      then paste the token into that file.
#   3. Copy this script to ~/bin/gce_trigger_forecast.sh and chmod +x it.
#   4. crontab -e, and add (the VM's clock must be UTC: check with `date`):
#        30 4 * * * $HOME/bin/gce_trigger_forecast.sh >> $HOME/gce_trigger_forecast.log 2>&1
#        0 9  * * * $HOME/bin/gce_trigger_forecast.sh >> $HOME/gce_trigger_forecast.log 2>&1

set -euo pipefail

REPO="AmedeeRoy/defile-migration-forecast"
WORKFLOW="predict_and_deploy_forecasts.yml"
TOKEN_FILE="${TOKEN_FILE:-$HOME/.config/defile/github_token}"

status=$(curl -sS -o /tmp/gce_trigger_forecast.out -w '%{http_code}' -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $(cat "$TOKEN_FILE")" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/$REPO/actions/workflows/$WORKFLOW/dispatches" \
  -d '{"ref":"main"}')

# 204 = dispatched; anything else (expired token, renamed workflow) is logged with the body.
if [ "$status" = "204" ]; then
  echo "$(date -u '+%F %T') dispatched"
else
  echo "$(date -u '+%F %T') FAILED HTTP $status: $(cat /tmp/gce_trigger_forecast.out)"
  exit 1
fi
