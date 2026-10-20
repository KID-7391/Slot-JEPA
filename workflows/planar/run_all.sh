#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --multi_gpu scripts/train.py --config-name train_PushObj
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --multi_gpu scripts/train.py --config-name train_PushT
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --multi_gpu scripts/train.py --config-name train_TwoRoomR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --multi_gpu scripts/train.py --config-name train_MazeR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --multi_gpu scripts/train.py --config-name train_TwoRoom
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --multi_gpu scripts/train.py --config-name train_Maze
