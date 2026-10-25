# Expected answers (written before asking the model)

1) a) sbatch header, 1 node, 8 CPUs, 16G, 2h, name, output with job id:

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=myjob_%j.out
```
Required flags: --nodes=1, 8 CPUs (--cpus-per-task=8 or --ntasks=8), --mem=16G, --time=02:00:00, --job-name, --output with %j.

2) b) array 1-10, max 3 concurrent:

```bash
#SBATCH --array=1-10%3
```
The %3 throttle is the required part.

3) c) dependency after success:

```bash
sbatch --dependency=afterok:42 B.sh
```
afterok:42 is the required part.

4) d) mail:

```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=me@example.com
```
Any correct mail-type set covering end and failure (END,FAIL or ALL) is fine.

5) e) srun 4 tasks:

```bash
srun --ntasks=4 hostname
```
(-n4 is an accepted short form.)

6) f) sacct elapsed and state:

```bash
sacct -j 99 --format=JobID,Elapsed,State
```
Note: on a cluster without accounting this fails with an error; record the real error.

7) g) full job configuration:

```bash
scontrol show job 77
```

8) h) sinfo partition, node, state:

```bash
sinfo -o "%P %N %T"
```
(-N -o "%N %T" without partition is also acceptable if partitions are not requested.)

9) i) hold and release:

```bash
scontrol hold 55
scontrol release 55
```

10) j) module + python via srun:

```bash
#!/bin/bash
#SBATCH --job-name=modpy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
module load python
srun python -c 'print(42)'
```
The container has no `module` command, so the real run must fail with "module: command not found". The script structure is what gets rated.
