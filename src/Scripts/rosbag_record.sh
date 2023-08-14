#==========================================
# Title: ROSBAG AUTO RECORD - Infrastructure
# Author: Neel Bhatt
# Date:   27 Jul 2020
#==========================================

# Directory setup
FOLDER_NAME=$(date +"%b%d-20%y_CAM-LCR_LIDAR_applanix")
SUB_FOLDER_NAME=$(date +"%H-%M-%S")

# Check if nvme drive at /dev/nvme0n1 is not mounted and if so mount it
# if !(grep -qs '/media/jetson/05ba3dfb-e1a0-471d-a378-3ac0444ba5bb' /proc/mounts)
# then
#     echo "Nvme drive not mounted! Mounting ..."
#     udisksctl mount -b /dev/nvme0n1
# fi

# Move to directory
cd /media/c66tang/Data/bagfiles;

# Prompt the user to enter some description of the test to be recorded
IFS= read -p "Enter brief description of the current test to be recorded: `echo '\n>'`" TEST_DESCRIPTION

mkdir -p $FOLDER_NAME/$SUB_FOLDER_NAME;
cd $FOLDER_NAME/$SUB_FOLDER_NAME;

# Specify topics to record (NOTE THE SPACES BETWEEN TOPICS)
 
TOPICS_TO_RECORD="/pylon_camera_node_center/image_rect/compressed\
 /rslidar_points_front\
 /tf
 /bbox_array/
 /rslidar_points_front/ground_filtered"

# Save test description and topic list to file (sed replaces spaces to newline character. (\) is autoremoved when saved to variable)
echo "TEST_DESCRIPTION:\n"$TEST_DESCRIPTION > test_description_and_topics.txt; 
echo "\nTOPICS: "$TOPICS_TO_RECORD | sed "s/[ ]/\n/g" >> test_description_and_topics.txt;
# Record topics (--buffsize=0)
rosbag record --split --duration 60 $TOPICS_TO_RECORD
exec bash