# Evaluation source bundle: multi constraint

## Source excerpt: slurm_man_sbatch#0

## Name

sbatch - Submit a batch script to Slurm.

## Synopsis

sbatch [OPTIONS(0)...] [ : [OPTIONS(N)...]] script(0) [args(0)...]

Option(s) define multiple jobs in a co-scheduled heterogeneous job.
For more details about heterogeneous jobs see the document

https://slurm.schedmd.com/heterogeneous_jobs.html

## Description

sbatch submits a batch script to Slurm. The batch script may be given to
sbatch through a file name on the command line, or if no file name is specified,
sbatch will read in a script from standard input.

The batch script may contain one or more lines beginning with "#SBATCH" followed
by any of the CLI options documented on this page. #SBATCH directives are read
directly by Slurm, so shell-specific syntax including variable names will be
read as literal text. Once the first non-comment, non-whitespace line has been
reached in the script, no more #SBATCH directives will be processed. See example
below.

sbatch exits immediately after the script is successfully transferred to the
Slurm controller and assigned a Slurm job ID. The batch script is not
necessarily granted resources immediately, it may sit in the queue of pending
jobs for some time before its required resources become available.

## Source excerpt: slurm_man_sbatch#2

**-A, --account=<account>**
Charge resources used by this job to specified account.
The account is an arbitrary string. The account name may
be changed after job submission using the scontrol
command.

**--acctg-freq=<datatype>=<interval>[,<datatype>=<interval>...]**
Define the job accounting and profiling sampling intervals in seconds.
This can be used to override the JobAcctGatherFrequency parameter in
the slurm.conf file. <datatype>=<interval> specifies the task
sampling interval for the jobacct_gather plugin or a
sampling interval for a profiling type by the
acct_gather_profile plugin. Multiple
comma-separated <datatype>=<interval> pairs
may be specified. Supported datatype values are:

**task**
Sampling interval for the jobacct_gather plugins and for task
profiling by the acct_gather_profile plugin.
NOTE: This frequency is used to monitor memory usage. If memory limits
are enforced, the highest frequency a user can request is what is configured
in the slurm.conf file. It can not be disabled.

**energy**
Sampling interval for energy profiling using the
acct_gather_energy plugin.

**network**
Sampling interval for infiniband profiling using the
acct_gather_interconnect plugin.

**filesystem**
Sampling interval for filesystem profiling using the
acct_gather_filesystem plugin.

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

## Source excerpt: slurm_man_sbatch#64

**--test-only**
Validate the batch script and return an estimate of when a job would be
scheduled to run given the current job queue and all the other arguments
specifying the job requirements. No job is actually submitted.

**--thread-spec=<num>**
Count of specialized threads per node reserved by the job for system operations
and not used by the application. The application will not use these threads,
but will be charged for their allocation.
This option can not be used with the --core-spec option.
NOTE: Explicitly setting a job's specialized thread value implicitly sets
its --exclusive option, reserving entire nodes for the job.

**--threads-per-core=<threads>**
Restrict node selection to nodes with at least the specified number of
threads per core. In task layout, use the specified maximum number of threads
per core. NOTE: "Threads" refers to the number of processing units on
each core rather than the number of application tasks to be launched per core.
See additional information under -B option above when task/affinity
plugin is enabled.
NOTE: This option may implicitly set the number of tasks (if -n
was not specified) as one task per requested thread.

## Source excerpt: slurm_man_sbatch#65

**--time-min=<time>**
Set a minimum time limit on the job allocation.
If specified, the job may have its --time limit lowered to a value
no lower than --time-min if doing so permits the job to begin
execution earlier than otherwise possible.
The job's time limit will not be changed after the job is allocated resources.
This is performed by a backfill scheduling algorithm to allocate resources
otherwise reserved for higher priority jobs.
Acceptable time formats include "minutes", "minutes:seconds",
"hours:minutes:seconds", "days-hours", "days-hours:minutes" and
"days-hours:minutes:seconds".

**--tmp=<size>[units]**
Specify a minimum amount of temporary disk space per node.
Default units are mebibytes.
Different units can be specified using the suffix [K|M|G|T].

**--tres-bind=<tres>:[verbose,]<type>[+<tres>:**
[verbose,]<type>...]
Specify a list of tres with their task binding options. Currently gres are the
only supported tres for this options. Specify gres as "gres/<gres_name>"
(e.g. gres/gpu)
Example: --tres-bind=gres/gpu:verbose,map:0,1,2,3+gres/nic:closest
By default, most tres are not bound to individual tasks
Supported binding type options for gres:

**closest**
Bind each task to the gres(s) which are closest.
In a NUMA environment, each task may be bound to more than one gres (i.e.
all gres in that NUMA environment).
