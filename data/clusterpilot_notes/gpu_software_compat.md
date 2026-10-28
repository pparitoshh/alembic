# GPU software versions: driver, CUDA, PyTorch wheels and containers

- The NVIDIA driver sets the newest CUDA version a program can use; `nvidia-smi` shows it as "CUDA Version". Users
  cannot update the driver on a cluster. Leonardo: driver 535.54.03, CUDA 12.2 (A100 64 GB).
- "The NVIDIA driver on your system is too old" from PyTorch means the installed wheel was built for a newer CUDA
  than the driver supports. Install a build for an older CUDA from the PyTorch index, for example
  `pip install torch --index-url https://download.pytorch.org/whl/cu121`, and install vLLM (or other GPU packages)
  in a version built for the same CUDA major version. Check on a compute node, not on a login node without GPUs:
  `python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"`.
- Errors such as "device kernel image is invalid" or "the provided PTX was compiled with an unsupported toolchain"
  appear when kernels are compiled at run time (Triton, FlashInfer, other JIT code) by a newer CUDA compiler than
  the driver supports. Use package builds that match the driver's CUDA version, or switch off the optional
  JIT-compiled component, and test the new code path once on a compute node.
- Containers (Singularity/Apptainer): pull on a login node, since compute nodes usually have no internet, and keep
  the image in $WORK or $SCRATCH: `singularity pull pytorch.sif docker://nvcr.io/nvidia/pytorch:<tag>`. Check the
  tag's release notes for the driver it needs. Run with GPUs: `srun singularity exec --nv pytorch.sif <command>`;
  `-B $WORK` makes a directory visible inside. Docker itself is not available on most clusters.
- AMD GPUs (LUMI-G: MI250X) do not run CUDA programs: use ROCm builds (PyTorch ROCm wheels or the site's
  containers), and HIP to port CUDA code.
