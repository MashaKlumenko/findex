#!/bin/sh
# Render, Fly, and Railway inject PORT. Local docker run keeps 8000.
set -eu
port="${PORT:-8000}"
exec uvicorn findex.web:app \
  --host 0.0.0.0 \
  --port "$port" \
  --proxy-headers \
  --forwarded-allow-ips="*"
