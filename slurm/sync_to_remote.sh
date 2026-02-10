#!/bin/bash
# ============================================
# Sync project to remote cluster
# Usage: ./slurm/sync_to_remote.sh username hostname [remote_dir]
# ============================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 <username> <hostname> [remote_dir]"
    echo "Example: $0 johndoe cluster.cs.university.edu"
    echo "Example: $0 johndoe cluster.cs.university.edu /scratch/johndoe/LLM"
    exit 1
fi

REMOTE_USER="$1"
REMOTE_HOST="$2"
REMOTE_DIR="${3:-~/LLM}"

echo "========================================"
echo "Syncing to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
echo "========================================"

# Get the project root (parent of slurm/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR" || exit 1

# Create remote directory first
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

# Sync project files
rsync -avz --progress \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude 'models/*' \
    --exclude 'logs/*.out' \
    --exclude 'logs/*.err' \
    --exclude '.cache' \
    --exclude '*.egg-info' \
    ./ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo ""
echo "========================================"
echo "Sync complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. SSH to remote: ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  2. Setup environment: cd ${REMOTE_DIR} && bash slurm/setup_remote.sh"
echo "  3. Submit job: sbatch slurm/train_7b_8x2080ti.sh"
echo ""
