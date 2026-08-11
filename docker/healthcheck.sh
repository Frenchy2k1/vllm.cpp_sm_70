#!/bin/bash
# Container HEALTHCHECK for the vllm.cpp server image.
#
# Deliberately dependency-free: bash's /dev/tcp instead of curl, so the runtime
# stage carries no HTTP client it would otherwise never use. The endpoint is
# /health (src/vllm/entrypoints/openai/api_server.cpp:998) and the server binds
# 0.0.0.0:8000 by default (server_main.cpp:135-136).
set -u

host=${VLLM_HEALTHCHECK_HOST:-127.0.0.1}
port=${VLLM_HEALTHCHECK_PORT:-8000}

exec 3<>"/dev/tcp/${host}/${port}" || exit 1
printf 'GET /health HTTP/1.1\r\nHost: %s:%s\r\nConnection: close\r\n\r\n' \
  "${host}" "${port}" >&3 || exit 1
read -r status_line <&3 || exit 1

case ${status_line} in
  *' 200 '*) exit 0 ;;
esac
exit 1
