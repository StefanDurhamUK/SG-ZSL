#!/usr/bin/env bash

# shellcheck disable=SC2164
cd ../libs

# shellcheck disable=SC2034
# hyper-parameters
noise_l1=100
noise_l2=300
noise_l3=500
nlw1=1
nlw2=0.5
nlw3=0.1
nlw4=10
kl='kl_loss'
eg=30

########################################################################################################
####  white box
### GZSL_all
## parameter: --noiseLen & --new_loss_weight
python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=GZSL_all
python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=GZSL_all
python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=GZSL_all
python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=GZSL_all
python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=GZSL_all
python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=GZSL_all
python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=GZSL_all
python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=GZSL_all
python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=GZSL_all
python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=GZSL_all
python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=GZSL_all
python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=GZSL_all


####  black box
### GZSL_all
## parameter: --noiseLen & --new_loss_weight
#python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#
#python3 main.py  --outSizeTS=50 --dataset=AWA2 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA2 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA2 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA2 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA2 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA2 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA2 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA2 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA2 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA2 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA2 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=50 --dataset=AWA2 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all

#python3 main.py  --outSizeTS=32 --dataset=APY --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=32 --dataset=APY --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=32 --dataset=APY --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=32 --dataset=APY --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=32 --dataset=APY --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=32 --dataset=APY --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=32 --dataset=APY --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=32 --dataset=APY --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=32 --dataset=APY --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=32 --dataset=APY --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=32 --dataset=APY --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all
#python3 main.py  --outSizeTS=32 --dataset=APY --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=GZSL_all



### AZSL-gzsl
## parameter: --noiseLen & --new_loss_weight
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl

#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=gzsl

#### AZSL-zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box  --task_categories=AZSL --AZSL_test=zsl
#
#
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=black_box --task_categories=AZSL --AZSL_test=zsl
#

### White-box
### AZSL-gzsl
### parameter: --noiseLen & --new_loss_weight
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA1 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=40 --outSizeZ=50  --dataset=AWA2 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#python3 main.py  --outSizeTS=20 --outSizeZ=32  --dataset=APY --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=gzsl
#
#### AZSL-zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA1 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#python3 main.py   --outSizeTS=40 --dataset=AWA2 --outSizeZ=10 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box  --task_categories=AZSL --AZSL_test=zsl
#
#
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l1 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l2 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw1 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw2 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw3 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=AZSL --AZSL_test=zsl
#python3 main.py  --outSizeTS=20 --dataset=APY --outSizeZ=12 --noiseLen=$noise_l3 --loss_type=$kl --new_loss_weight=$nlw4 --epochG=$eg --n_samples=400 --n_z_samples=600   --framework=white_box --task_categories=AZSL --AZSL_test=zsl

