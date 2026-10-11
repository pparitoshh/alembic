# Evaluation source bundle: resources

## Source excerpt: slurm_man_sbatch#16

**-c, --cpus-per-task=<ncpus>**
Advise the Slurm controller that ensuing job steps will require ncpus
number of processors per task. Without this option, the controller will
just try to allocate one processor per task.
For instance,
consider an application that has 4 tasks, each requiring 3 processors. If our
cluster is comprised of quad-processors nodes and we simply ask for
12 processors, the controller might give us only 3 nodes. However, by using
the --cpus-per-task=3 options, the controller knows that each task requires
3 processors on the same node, and the controller will grant an allocation
of 4 nodes, one for each of the 4 tasks.

**--deadline=<OPT>**
Remove the job if no ending is possible before
this deadline (start > (deadline - time[-min])).
Default is no deadline. Note that if neither DefaultTime nor
MaxTime are configured on the partition the job is in, the job will
need to specify some form of time limit (--time[-min]) if a deadline
is to be used.
Valid time formats are:
HH:MM[:SS] [AM|PM]
MMDD[YY] or MM/DD[/YY] or MM.DD[.YY]
MM/DD[/YY]-HH:MM[:SS]
YYYY-MM-DD[THH:MM[:SS]]
now[+count[seconds(default)|minutes|hours|days|weeks]]
midnight, elevenses (11 AM), noon, fika (3 PM), teatime (4 PM), or tomorrow
One or more time strings may be specified (e.g., 'tomorrow18:00'). If there is
a conflict between them, the last one will silently take precedence.

## Source excerpt: slurm_man_sbatch#32

**-G, --gpus=[type:]<number>**
Specify the total number of GPUs required for the job.
An optional GPU type specification can be supplied.
For example "--gpus=volta:3".
See also the --gpus-per-node, --gpus-per-socket and
--gpus-per-task options.
NOTE: The allocation has to contain at least one GPU per node, or one of
each GPU type per node if types are used. Use heterogeneous jobs if different
nodes need different GPU types.

**--gpus-per-node=[type:]<number>**
Specify the number of GPUs required for the job on each node included in
the job's resource allocation.
An optional GPU type specification can be supplied.
For example "--gpus-per-node=volta:3".
Multiple options can be requested in a comma separated list, for example:
"--gpus-per-node=volta:3,kepler:1".
See also the --gpus, --gpus-per-socket and
--gpus-per-task options.
NOTE: This option is mutually exclusive with --gres=gpu.

**--gpus-per-socket=[type:]<number>**
Specify the number of GPUs required for the job on each socket included in
the job's resource allocation.
An optional GPU type specification can be supplied.
For example "--gpus-per-socket=volta:3".
Multiple options can be requested in a comma separated list, for example:
"--gpus-per-socket=volta:3,kepler:1".
Requires job to specify a sockets per node count ( --sockets-per-node).
See also the --gpus, --gpus-per-node and
--gpus-per-task options.

## Source excerpt: slurm_man_sbatch#33

**--gpus-per-task=[type:]<number>**
Specify the number of GPUs required for the job on each task to be spawned
in the job's resource allocation.
An optional GPU type specification can be supplied.
For example "--gpus-per-task=volta:1". Multiple options can be
requested in a comma separated list, for example:
"--gpus-per-task=volta:3,kepler:1". See also the --gpus,
--gpus-per-socket and --gpus-per-node options.
This option requires an explicit task count, e.g. -n, --ntasks or "--gpus=X
--gpus-per-task=Y" rather than an ambiguous range of nodes with -N, --nodes.
This option will implicitly set --tres-bind=gres/gpu:per_task:<gpus_per_task>,
or if multiple gpu types are specified
--tres-bind=gres/gpu:per_task:<gpus_per_task_type_sum>. However, that can be
overridden with an explicit --tres-bind=gres/gpu specification.

## Source excerpt: slurm_man_sbatch#34

**--gres=<list>**
Specifies a comma-delimited list of generic consumable resources requested per
node.
The format for each entry in the list is "name[[:type]:count]".
The name is the type of consumable resource (e.g. gpu).
The type is an optional classification for the resource (e.g. a100).
The count is the number of those resources with a default value of 1.
The count can have a suffix of
"k" or "K" (multiple of 1024),
"m" or "M" (multiple of 1024 x 1024),
"g" or "G" (multiple of 1024 x 1024 x 1024),
"t" or "T" (multiple of 1024 x 1024 x 1024 x 1024),
"p" or "P" (multiple of 1024 x 1024 x 1024 x 1024 x 1024).
The specified resources will be allocated to the job on each node.
The available generic consumable resources is configurable by the system
administrator.
A list of available generic consumable resources will be printed and the
command will exit if the option argument is "help".
Examples of use include "--gres=gpu:2", "--gres=gpu:kepler:2", and
"--gres=help".
NOTE: This option is mutually exclusive with --gpus-per-node.

**--gres-flags=<type>**
Specify generic resource task binding options.

**multiple-tasks-per-sharing**
Negate one-task-per-sharing. This is useful if it is set by default in
SelectTypeParameters.

**disable-binding**
Negate enforce-binding. This is useful if it is set by default in
SelectTypeParameters.

## Source excerpt: slurm_man_sbatch#40

**--mem=<size>[units]**
Specify the real memory required per node.
Default units are mebibytes.
Different units can be specified using the suffix [K|M|G|T].
Default value is DefMemPerNode and the maximum value is
MaxMemPerNode. If configured, both parameters can be
seen using the scontrol show config command.
This parameter would generally be used if whole nodes
are allocated to jobs (SelectType=select/linear).
Also see --mem-per-cpu and --mem-per-gpu.
The --mem, --mem-per-cpu and --mem-per-gpu
options are mutually exclusive. If --mem, --mem-per-cpu or
--mem-per-gpu are specified as command line arguments, then they will
take precedence over the environment.
NOTE: A memory size specification of zero is treated as a special case and
grants the job access to all of the memory on each node.
NOTE: The memory used by each slurmstepd process is included in the job's
total memory usage. It typically consumes between 20MiB and 200MiB, though this
can vary depending on system configuration and any loaded plugins.
NOTE: Memory requests will not be strictly enforced unless Slurm is
configured to use an enforcement mechanism. See ConstrainRAMSpace in
the cgroup.conf(5) man page and OverMemoryKill in the
slurm.conf(5) man page for more details.

## Source excerpt: slurm_man_sbatch#44

**--mem-per-cpu=<size>[units]**
Minimum memory required per usable allocated CPU.
Default units are mebibytes.
The default value is DefMemPerCPU and the maximum value is
MaxMemPerCPU (see exception below). If configured, both parameters can be
seen using the scontrol show config command.
Note that if the job's --mem-per-cpu value exceeds the configured
MaxMemPerCPU, then the user's limit will be treated as a memory limit
per task; --mem-per-cpu will be reduced to a value no larger than
MaxMemPerCPU; --cpus-per-task will be set and the value of
--cpus-per-task multiplied by the new --mem-per-cpu
value will equal the original --mem-per-cpu value specified by
the user.
This parameter would generally be used if individual processors
are allocated to jobs (SelectType=select/cons_tres).
If resources are allocated by core, socket, or whole nodes, then the number
of CPUs allocated to a job may be higher than the task count and the value
of --mem-per-cpu should be adjusted accordingly.
Also see --mem and --mem-per-gpu.
The --mem, --mem-per-cpu and --mem-per-gpu
options are mutually exclusive.
NOTE: If the final amount of memory requested by a job
can't be satisfied by any of the nodes configured in the
partition, the job will be rejected.
This could happen if --mem-per-cpu is used with the
--exclusive option for a job allocation and --mem-per-cpu
times the number of CPUs on a node is greater than the total memory of that
node.
NOTE: This applies to usable allocated CPUs in a job allocation.
This is important when more than one thread per core is configured.
If a job requests --threads-per-core with fewer threads on a core than
exist on the core (or --hint=nomultithread which implies
--threads-per-core=1), the job will be unable to use those extra threads on
the core and those threads will not be included in the memory per CPU
calculation. But if the job has access to all threads on the core, those threads
will be included in the memory per CPU calculation even if the job did not
explicitly request those threads.
In the following examples, each core has two threads.
In this first example, two tasks can run on separate hyperthreads
in the same core because --threads-per-core is not used. The
third task uses both threads of the second core. The allocated
memory per cpu includes all threads:

## Source excerpt: slurm_man_sbatch#46

**--mem-per-gpu=<size>[units]**
Minimum memory required per allocated GPU.
Default units are mebibytes.
Different units can be specified using the suffix [K|M|G|T].
Default value is DefMemPerGPU and is available on both a global and
per partition basis.
If configured, the parameters can be seen using the scontrol show config
and scontrol show partition commands.
Also see --mem.
The --mem, --mem-per-cpu and --mem-per-gpu
options are mutually exclusive.

**--mincpus=<n>**
Specify a minimum number of logical cpus/processors per node.

**--network=<type>**
Specify information pertaining to the switch or network.
The interpretation of type is system dependent.
The network option is available on systems with HPE Slingshot
networks. It can be used to request a job VNI (to be used for communication
between job steps in a job). It also can be used to override the default
network resources allocated for the job step. Multiple values may be specified
in a comma-separated list.

**tcs=<class1>[:<class2>]...**
Set of traffic classes to configure for applications.
Supported traffic classes are DEDICATED_ACCESS, LOW_LATENCY, BULK_DATA, and
BEST_EFFORT. The traffic classes may also be specified as TC_DEDICATED_ACCESS,
TC_LOW_LATENCY, TC_BULK_DATA, and TC_BEST_EFFORT.

**no_vni**
Don't allocate any VNIs for this job (even if multi-node).

**job_vni**
Allocate a job VNI for this job.

**single_node_vni**
Allocate a job VNI for this job, even if it is a single-node job.

## Source excerpt: slurm_man_sbatch#51

**-N, --nodes=<minnodes>[-maxnodes]|<size_string>**
Request that a minimum of minnodes nodes be allocated to this job.
A maximum node count may also be specified with maxnodes.
If only one number is specified, this is used as both the minimum and
maximum node count. Node count can be also specified as size_string.
The size_string specification identifies what nodes values should be used.
Multiple values may be specified using a comma separated list or
with a step function by suffix containing a colon and
number values with a "-" separator.
For example, "--nodes=1-15:4" is equivalent to "--nodes=1,5,9,13".
The partition's node limits supersede those of the job.
If a job's node limits are outside of the range permitted for its
associated partition, the job will be left in a PENDING state.
This permits possible execution at a later time, when the partition
limit is changed.
If a job node limit exceeds the number of nodes configured in the
partition, the job will be rejected.
Note that the environment
variable SLURM_JOB_NUM_NODES will be set to the count of nodes actually
allocated to the job. See the ENVIRONMENT VARIABLES  section
for more information. If -N is not specified, the default
behavior is to allocate enough nodes to satisfy the requested resources as
expressed by per-job specification options, e.g. -n, -c and
--gpus.
The job will be allocated as many nodes as possible within the range specified
and without delaying the initiation of the job.
The node count specification may include a numeric value followed by a suffix
of "k" (multiplies numeric value by 1,024) or "m" (multiplies numeric value by
1,048,576).
NOTE: This option cannot be used in with arbitrary distribution.

## Source excerpt: slurm_man_sbatch#52

**-n, --ntasks=<number>**
sbatch does not launch tasks, it requests an allocation of resources and
submits a batch script. This option advises the Slurm controller that job
steps run within the allocation will launch a maximum of number
tasks and to provide for sufficient resources.
The default is one task per node, but note
that the --cpus-per-task option will change this default.

**--ntasks-per-core=<ntasks>**
Request the maximum ntasks be invoked on each core.
Meant to be used with the --ntasks option.
Related to --ntasks-per-node except at the core level
instead of the node level. This option will be inherited by srun.
Slurm may allocate more cpus than what was requested in order to respect this
option.
NOTE: This option is not supported when using
SelectType=select/linear. This value can not be greater than
--threads-per-core.

## Source excerpt: slurm_man_sbatch#54

**--ntasks-per-node=<ntasks>**
Request that ntasks be invoked on each node.
If used with the --ntasks option, the --ntasks option will take
precedence and the --ntasks-per-node will be treated as a
maximum count of tasks per node.
Meant to be used with the --nodes option.
This is related to --cpus-per-task=ncpus,
but does not require knowledge of the actual number of cpus on
each node. In some cases, it is more convenient to be able to
request that no more than a specific number of tasks be invoked
on each node. Examples of this include submitting
a hybrid MPI/OpenMP app where only one MPI "task/rank" should be
assigned to each node while allowing the OpenMP portion to utilize
all of the parallelism present in the node, or submitting a single
setup/cleanup/monitoring job to each node of a pre-existing
allocation as one step in a larger job script.

**--ntasks-per-socket=<ntasks>**
Request the maximum ntasks be invoked on each socket.
Meant to be used with the --ntasks option.
Related to --ntasks-per-node except at the socket level
instead of the node level.
NOTE: This option is not supported when using
SelectType=select/linear.
