# constants
CONDA_ENV_NAME=tas

MAIN_HOME=/home
MAIN_USER=s1915791
MAIN_PROJECT=git/robust-perception-tas
MAIN_PATH=${MAIN_HOME}/${MAIN_USER}
MAIN_PROJECT_PATH=${MAIN_PATH}/${MAIN_PROJECT}

SCRATCH_HOME=/disk/scratch
SCRATCH_USER=s1915791
SCRATCH_PROJECT=git/robust-perception-tas
SCRATCH_PATH=${SCRATCH_HOME}/${SCRATCH_USER}
SCRATCH_PROJECT_PATH=${SCRATCH_PATH}/${SCRATCH_PROJECT}

DATA_DN=data
OUTPUT_DN=ckpt
INPUT_PATH=${DATA_DN}/sets

DATA_SCRIPT_FN=pfp_odgt.py


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
mkdir -p ${SCRATCH_PATH}

# Activate your conda environment
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}


# =================================
# Move input data to scratch disk
# =================================
# Move data from a source location, probably on the distributed filesystem
# (DFS), to the scratch space on the selected node. Your code should read and
# write data on the scratch space attached directly to the compute node (i.e.
# not distributed), *not* the DFS. Writing/reading from the DFS is extremely
# slow because the data must stay consistent on *all* nodes. This constraint
# results in much network traffic and waiting time for you!
#
# This example assumes you have a folder containing all your input data on the
# DFS, and it copies all that data  file to the scratch space, and unzips it. 
#
# For more guidelines about moving files between the distributed filesystem and
# the scratch space on the nodes, see:
#     http://computing.help.inf.ed.ac.uk/cluster-tips

echo "Moving input data to the compute node's scratch space: $SCRATCH_HOME"

# input data directory path on the DFS
src_path=${MAIN_PROJECT_PATH}/${INPUT_PATH}


# input data directory path on the scratch disk of the node
dest_path=${SCRATCH_PROJECT_PATH}/${INPUT_PATH}
mkdir -p ${dest_path}  # make it if required

# Important notes about rsync:
# * the --compress option is going to compress the data before transfer to send
#   as a stream. THIS IS IMPORTANT - transferring many files is very very slow
# * the final slash at the end of ${src_path}/ is important if you want to send
#   its contents, rather than the directory itself. For example, without a
#   final slash here, we would create an extra directory at the destination:
#       ${SCRATCH_HOME}/project_name/data/input/input
# * for more about the (endless) rsync options, see the docs:
#       https://download.samba.org/pub/rsync/rsync.html

rsync --archive --update --compress --progress ${src_path}/ ${dest_path}


# ======================
# Pre-processing data
# ======================
# pre-processing the data on the scratch disk with the data script

echo "Pre-processing data in scratch space"

python ${MAIN_PROJECT_PATH}/${DATA_DN}/${DATA_SCRIPT_FN} --dir ${SCRATCH_PROJECT_PATH}/${INPUT_PATH}


# ==============================
# Finally, run the experiment!
# ==============================
python ${MAIN_PROJECT_PATH}/train.py -c ${MAIN_PROJECT_PATH}/config/retinanet_resnet50_fpn-pennfudanped.yaml -i ${SCRATCH_PROJECT_PATH}/${INPUT_PATH}/PennFudanPed -o ${SCRATCH_PROJECT_PATH}/ckpt/retinanet_resnet50_fpn-pennfudanped TRAIN.DATA.batch_size 1

# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"

src_path=${SCRATCH_PROJECT_PATH}/${OUTPUT_DN}
dest_path=${MAIN_PROJECT_PATH}/${OUTPUT_DN}
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "Job finished successfully!"