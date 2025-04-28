#!/bin/bash
### Sets the job's name.
#SBATCH --job-name=myFirstJob

### Sets the job's output file and path.
#SBATCH --output=myFirstJob.out.%j

### Sets the job's error output file and path.
#SBTACH --error=myFirstJob.err.%j

### Requested number of nodes for this job. Can be a single number or a range.
#SBATCH -N 1

### Requested partition (group of nodes, i.e. compute, bigmem, gpu, etc.) for the resource allocation.
#SBATCH -p kimq

### Requested number of gpus
#SBATCH --gres=gpu:1

### Limit on the total run time of the job allocation.
#SBATCH --time=1:00:00

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "Activating TensorFlow-2.6.2 environment"
source /shared/tensorflow-2.6.2/tf_env/bin/activate

echo "Running testTF.py"
python3 ~/testTFForSlurm/testTF.py

echo "Deactivating TensorFlow-2.6.2 environment"
deactivate

echo "Done."
