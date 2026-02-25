#!/bin/bash
#SBATCH --job-name=sgemm_kokkos
#SBATCH --output=sgemm_kokkos_%j.out
#SBATCH --error=sgemm_kokkos_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=cin_staff
#SBATCH --cpus-per-task=4
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:15:00

KERNEL_NUM=${1:-1}

echo "============================================"
echo "  SGEMM Kokkos - Kernel ${KERNEL_NUM}"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "============================================"

module load profile/candidate
module load nvhpc/24.5
module load gcc/12.2.0

# Eventually move this to a CMakePresets.json file
BUILD_DIR="build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}" && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
cd ..

echo ""
echo ">>> Running sgemm kernel ${KERNEL_NUM} ..."
./${BUILD_DIR}/sgemm "${KERNEL_NUM}"
