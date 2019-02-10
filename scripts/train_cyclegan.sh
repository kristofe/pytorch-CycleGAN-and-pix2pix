#python train.py --dataroot ./datasets/terrain_2560_small/aligned/ --name terrain_cyclegan --model cycle_gan --pool_size 50 --align_data

python train.py --dataroot ./datasets_demo/256/curr_topo_3k --name terrain_cyclegan_curr_topo --which_model_netG unet_256 --model cycle_gan --pool_size 50 --gpu_ids 1