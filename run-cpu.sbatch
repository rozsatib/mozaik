#!/bin/bash

#SBATCH --account=nix00014
#SBATCH --job-name=compare             # Name of the job
#SBATCH --ntasks=12                                  # Number of tasks
#SBATCH --cpus-per-task=4                                       # Number of CPU cores per task
#SBATCH --nodes 1                                   # Ensure that all cores are on one machine
#SBATCH --time=0-48:00                              # Runtime in D-HH:MM
#SBATCH --mem 128G                                  # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=./slurm_cpu/slurm-%x-%j.out              # File to which STDOUT will be written
#SBATCH --error=./slurm_cpu/slurm-%x-%j.err               # File to which STDERR will be written
#SBATCH --mail-type=ALL                             # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=goirik.chakrabarty@uni-goettingen.de    # Email to which notifications will be sent
#SBATCH -p standard96                              # Partition to submit to
#SBATCH -C cascadelake                             # Constraint for Cascade Lake nodes

export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export FTP_PROXY="http://www-cache.gwdg.de:3128"

module load apptainer

cd /mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/mozaik

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

echo "Unsetting proxies to avoid 'parameters' library bug..."
unset HTTP_PROXY
unset HTTPS_PROXY
unset FTP_PROXY

bash apptainer-compose.sh