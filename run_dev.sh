#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_ARGS=(-f docker-compose.yml -f docker-compose.dev.yml)

cleanup() {
  echo
  echo "Stopping services..."
  if [[ -n "${LOGS_PID:-}" ]]; then
    kill "$LOGS_PID" 2>/dev/null || true
  fi
  docker compose "${COMPOSE_ARGS[@]}" down
}
trap cleanup EXIT

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok не найден в PATH. Установите его и повторите." >&2
  exit 1
fi

docker compose "${COMPOSE_ARGS[@]}" up -d "$@"

echo "Docker контейнеры запущены. Логи приложения:" 
docker compose "${COMPOSE_ARGS[@]}" logs -f &
LOGS_PID=$!

sleep 2
echo
echo "Запускаю ngrok туннель (Ctrl+C для остановки):"
ngrok http 8000
