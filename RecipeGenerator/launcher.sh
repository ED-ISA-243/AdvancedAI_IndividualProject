#!/bin/sh
set -e
APP_RESOURCES="$(cd "$(dirname "$0")" && pwd)"

INNER="$APP_RESOURCES/run_inner.sh"
cat > "$INNER" <<'EOS'
#!/bin/sh
set -e
APP_RESOURCES="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_RESOURCES"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TRANSFORMERS_OFFLINE=1
exec "$APP_RESOURCES/venv/bin/python" "$APP_RESOURCES/app/gui.py"
EOS
chmod +x "$INNER"

/usr/bin/osascript <<OSA
tell application "Terminal"
    activate
    do script "bash \"${INNER}\""
end tell
OSA