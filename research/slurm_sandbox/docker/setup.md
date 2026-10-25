# Docker Slurm sandbox: exact setup steps

This lab ran in container `slurm-test` on ubuntu:24.04 with slurm-wlm 23.11.4.
All steps were used as shown on 2026-10-25. Nothing here is theoretical.

## Files in this folder

| File | Target in container | Purpose |
|---|---|---|
| slurm.conf | /etc/slurm/slurm.conf | cluster config, one node, two partitions |
| gres.conf | /etc/slurm/gres.conf | 4 fake GPUs on /dev/fakegpu0-3 |
| cgroup.conf | /etc/slurm/cgroup.conf | IgnoreSystemdOnFailure=yes |
| slurmdbd.conf | /etc/slurm/slurmdbd.conf | accounting DB config; throwaway password |
| start-slurmd.sh | /usr/local/bin/start-slurmd.sh | builds the fake cgroup tree, starts slurmd with the shim |
| cgroup_shim.c | /tmp/cgroup_shim.c, compiled to /usr/local/lib/libcgroup_shim.so | LD_PRELOAD shim for cgroup detection |

## 1. Container

```bash
docker run -d --name slurm-test --hostname slurmnode ubuntu:24.04 sleep infinity
```

No extra docker options worked here: the host cgroupfs is read-only in the
container and unshare is blocked. The shim (step 5) is the workaround.

## 2. Packages

```bash
docker exec slurm-test bash -c "apt-get update -qq && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  slurm-wlm munge gcc slurmdbd mariadb-server"
```

Installed: slurm-wlm 23.11.4, munge 0.5.15, gcc 13.3 (to compile the shim),
slurmdbd + mariadb-server (for accounting; see step 6).

## 3. Munge

```bash
docker exec slurm-test bash -c '
  mkdir -p /run/munge && chown munge:munge /run/munge
  chmod 700 /var/lib/munge /var/log/munge
  rm -f /tmp/munged.log && touch /tmp/munged.log
  chown munge:munge /tmp/munged.log
  su -s /bin/bash munge -c "/usr/sbin/munged --log-file /tmp/munged.log"
  munge -n | unmunge'
```

Notes from the real run:

- The package already creates /etc/munge/munge.key; do not run mungekey again.
- Starting munged as root fails twice: first "Logfile is insecure" (log file
  must be owned by root before the daemon drops privileges), then the PRNG
  seed dir check. Starting munged AS the munge user with its own log file
  works.

## 4. Slurm config

```bash
docker cp slurm.conf slurm-test:/etc/slurm/slurm.conf
docker cp gres.conf   slurm-test:/etc/slurm/gres.conf
docker cp cgroup.conf slurm-test:/etc/slurm/cgroup.conf
docker exec slurm-test bash -c '
  mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
  chown slurm:slurm /var/spool/slurmctld /var/log/slurm
  chmod 755 /var/spool/slurmd /var/log/slurm
  chmod 600 /etc/slurm/slurmdbd.conf
  for i in 0 1 2 3; do mknod /dev/fakegpu$i c 195 $i; done'
```

Fake GPUs: `Name=gpu File=/dev/null Count=4` fails ("duplicate device file
name"); a bare `Name=gpu Count=4` is ignored for gpu ("Ignoring file-less
GPU"). Four distinct char devices on major 195 work.

## 5. cgroup shim

Slurm 23.11 reads /proc/self/cgroup and fatals on "Hybrid mode is not
supported" because the host leaks a v1 net_cls line. The shim:

- serves /tmp/fake_cgroup.<pid> instead of /proc/<pid>/cgroup: root for
  slurmd, the stepd scope path for slurmstepd,
- maps /sys/fs/cgroup to the writable tree /tmp/cgfs,
- auto-creates the cgroup pseudo files in each new directory.

```bash
docker cp cgroup_shim.c slurm-test:/tmp/cgroup_shim.c
docker exec slurm-test bash -c '
  gcc -shared -fPIC -O2 -o /usr/local/lib/libcgroup_shim.so /tmp/cgroup_shim.c -ldl
  mkdir -p /tmp/cgfs/system.slice'
docker cp start-slurmd.sh slurm-test:/usr/local/bin/start-slurmd.sh
docker exec slurm-test chmod +x /usr/local/bin/start-slurmd.sh
```

## 6. Accounting (slurmdbd + MariaDB)

Without accounting, 23.11 pends jobs with reason InvalidAccount (the
accounting_storage/none plugin was removed in 23.11). MariaDB + slurmdbd fix
this.

```bash
docker exec slurm-test bash -c '
  mkdir -p /run/mysqld && chown mysql:mysql /run/mysqld
  (setsid mysqld_safe >/tmp/mysql.out 2>&1 &); sleep 8
  mysql -e "CREATE USER \"slurmdba\"@\"localhost\" IDENTIFIED BY \"testpass\";
            CREATE DATABASE slurm_acct_db;
            GRANT ALL ON slurm_acct_db.* TO \"slurmdba\"@\"localhost\";"
  chown slurm /etc/slurm/slurmdbd.conf && chmod 600 /etc/slurm/slurmdbd.conf
  su -s /bin/bash slurm -c "/usr/sbin/slurmdbd"'
```

The password in the repo copy is the placeholder CHANGEME; the container used
the throwaway testpass.

## 7. Start order

```
1. mariadbd          (mysqld_safe, needs /run/mysqld owned by mysql)
2. munged            (as munge user, own log file)
3. slurmdbd          (as slurm user)
4. slurmctld         (plain binary, daemonizes)
5. slurmd            (via start-slurmd.sh: shim + fake cgroup tree)
```

Then register the cluster in the DB once:

```bash
docker exec slurm-test sacctmgr -i add cluster slurm-test
```

## 8. slurm.conf facts learned on the way

- `priority/multifactor` (Ubuntu default) needs accounting; use
  `priority/basic` or provide slurmdbd.
- `RealMemory` in the node line must be <= what slurmd reports, else the
  node registration fails with "Low RealMemory". Either omit it or raise the
  threshold: `SlurmctldParameters=node_reg_mem_percent=40`.
- Omitting `CPUs=` defaults the node to 1 CPU. This host has 12, so the
  config pins CPUs=12 (declared >= the 8-CPU test jobs).
- slurmd must start through start-slurmd.sh (shim), or the cgroup/v2 plugin
  init fails and slurmd exits.
