#!/bin/bash
mkdir -p /tmp/cgfs/system.slice
printf "0::/\n" > /tmp/fake_cgroup
printf "cpuset cpu memory pids\n" > /tmp/cgfs/cgroup.controllers
for f in cgroup.subtree_control cgroup.procs cgroup.type cgroup.stat cgroup.events cgroup.threads cgroup.freeze cgroup.cpus.controllers cgroup.cpus.subtree_control cgroup.memory.controllers cgroup.memory.subtree_control; do
	[ -e /tmp/cgfs/$f ] || : > /tmp/cgfs/$f
done
exec env LD_PRELOAD=/usr/local/lib/libcgroup_shim.so /usr/sbin/slurmd -D "$@"
