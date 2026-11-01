# Common environment for the PowerSGD experiments on Perlmutter. Source it, don't execute it:
#   source perlmutter/env.sh
module load "${PYTORCH_MODULE:-pytorch}"

export REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export DATA=${DATA:-$SCRATCH/powersgd}
export RESULTS=${RESULTS:-$SCRATCH/powersgd/results}
export PYTHONPATH=$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}
