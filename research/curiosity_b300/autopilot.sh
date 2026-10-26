#!/bin/bash
# Runs autopilot.py every 2 minutes on the login node. Stop: touch ~/cp/autopilot/STOP
mkdir -p $HOME/cp/autopilot
while [ ! -e $HOME/cp/autopilot/STOP ]; do
    python3 $HOME/cp/bin/autopilot.py >> $HOME/cp/logs/autopilot.err 2>&1
    touch $HOME/cp/autopilot/HEARTBEAT
    sleep 120
done
