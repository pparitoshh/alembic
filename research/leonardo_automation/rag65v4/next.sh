#!/bin/bash
# Started by the round-4 controller after its scores.
exec "$(cd "$(dirname "$0")" && pwd)/../rag65v5/controller.sh"
