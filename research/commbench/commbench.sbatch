#!/bin/bash
#SBATCH --job-name=dk-commbench
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:25:00
#SBATCH --output=/leonardo_work/EUHPC_D30_031/runs/a08trc09/commbench/run-%j/job.out
# Node-to-node communication measurements on Leonardo Booster (Oct 27, MPI/NCCL data push): topology, OSU point-to-point
# and allreduce (host and GPU memory, CUDA-aware Open MPI), NCCL all-reduce under several NCCL settings. Each result
# goes to its own file in run-<jobid>/ and later becomes source text for grounded Q&A.
W=/leonardo_work/EUHPC_D30_031/runs/a08trc09/commbench; O=$W/run-$SLURM_JOB_ID; mkdir -p $O; cd $O
VPY=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
H=($(scontrol show hostnames $SLURM_JOB_NODELIST)); echo "nodes: ${H[*]}" > info.txt
{ nvidia-smi -L; nvidia-smi topo -m; ibv_devinfo -l 2>&1; ibstat 2>&1 | grep -E "CA |Rate|State|Link layer" ; lscpu | grep -E "Model name|Socket|NUMA node\(s\)|^CPU\(s\)"; } >> info.txt 2>&1
module load osu-micro-benchmarks/7.4--openmpi--4.1.6--gcc--12.2.0-cuda-12.2 > /dev/null 2>&1
{ ompi_info --version 2>&1 | head -2; ucx_info -v 2>&1 | head -3; } >> info.txt
P=$(dirname $(which osu_bw)); C=$P/../collective
run() { local f=$1; shift; echo "### $*" > $f; timeout 240 "$@" >> $f 2>&1; echo "### rc=$?" >> $f; }
run osu_bw_inter_host.txt   mpirun -np 2 --map-by ppr:1:node $P/osu_bw
run osu_bw_inter_gpu.txt    mpirun -np 2 --map-by ppr:1:node -x UCX_TLS $P/osu_bw -d cuda D D
run osu_lat_inter_host.txt  mpirun -np 2 --map-by ppr:1:node $P/osu_latency
run osu_lat_inter_gpu.txt   mpirun -np 2 --map-by ppr:1:node $P/osu_latency -d cuda D D
run osu_bibw_inter_gpu.txt  mpirun -np 2 --map-by ppr:1:node $P/osu_bibw -d cuda D D
run osu_bw_intra_gpu.txt    mpirun -np 2 --host ${H[0]}:2 $P/osu_bw -d cuda D D
run osu_allreduce_8gpu.txt  mpirun -np 8 --map-by ppr:4:node $C/osu_allreduce -d cuda -m 1048576:268435456
run osu_allreduce_8host.txt mpirun -np 8 --map-by ppr:4:node $C/osu_allreduce -m 1048576:268435456
module purge > /dev/null 2>&1
export MASTER_ADDR=${H[0]} MASTER_PORT=29511
nccl() { local f=$1; shift; echo "### env $*" > $f; timeout 240 srun --ntasks=8 --ntasks-per-node=4 env "$@" $VPY $W/nccl_bench.py >> $f 2>&1; echo "### rc=$?" >> $f; }
nccl nccl_default_debug.txt NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
nccl nccl_ring.txt          NCCL_ALGO=Ring
nccl nccl_tree.txt          NCCL_ALGO=Tree
nccl nccl_no_ib.txt         NCCL_IB_DISABLE=1
nccl nccl_no_gdr.txt        NCCL_NET_GDR_LEVEL=LOC
echo done > DONE
