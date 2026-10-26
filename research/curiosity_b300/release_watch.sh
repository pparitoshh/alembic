#!/bin/bash
# Laptop side of the Curiosity release: every 5 minutes, if the autopilot wrote ~/cp/release/READY for a model not yet
# published, build the release on Curiosity, then push it to Hugging Face with the token streamed over ssh (never
# stored on the cluster). Log: release_watch.log next to this script.
cd "$(dirname "$0")"
while [ ! -e STOP ]; do
    name=$(ssh -o ConnectTimeout=30 curiosity 'test -e ~/cp/release/READY && python3 -c "import json;print(json.load(open(\"$HOME/cp/release/READY\"))[\"name\"])"' 2>/dev/null)
    if [ -n "$name" ] && ! grep -q "^PUBLISHED $name$" release_watch.log 2>/dev/null; then
        echo "$(date -u +%FT%TZ) building $name" >> release_watch.log
        if ssh curiosity 'bash ~/cp/bin/release_build.sh' >> release_watch.log 2>&1; then
            if cat ~/.cache/huggingface/token | ssh curiosity "read -r T; HF_TOKEN=\$T ~/cp/venv/bin/python ~/cp/bin/hf_push.py ~/cp/release/$name" >> release_watch.log 2>&1; then
                echo "PUBLISHED $name" >> release_watch.log
            else echo "$(date -u +%FT%TZ) push failed for $name" >> release_watch.log; fi
        else echo "$(date -u +%FT%TZ) build failed for $name" >> release_watch.log; fi
    fi
    sleep 300
done
