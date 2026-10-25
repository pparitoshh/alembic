#!/bin/bash
# Started by the factory controller after the merge.
exec "$(cd "$(dirname "$0")" && pwd)/../rag65v4/controller.sh"
