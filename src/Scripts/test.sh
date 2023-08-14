#==========================================
# Title: ROSBAG AUTO RECORD - Infrastructure
# Author: Neel Bhatt
# Date:   27 Jul 2020
#==========================================

# Check if nvme drive at /dev/nvme0n1 is not mounted and if so mount it
if !(grep -qs '/media/jetson1/05ba3dfb-e1a0-471d-a378-3ac0444ba5bb' /proc/mounts)
then
    echo "Nvme drive not mounted! Mounting ..."
    udisksctl mount -b /dev/nvme0n1
fi
# Move to directory
cd /media/jetson1/05ba3dfb-e1a0-471d-a378-3ac0444ba5bb;
read -p "Enter brief description of the current test to be recorded: `echo '\n>'`" TEST_DESCRIPTION
echo "$TEST_DESCRIPTION"
#exec bash