python3 train_satellites.py \
	--arch linknet34 --batch-size 40 \
	--imsize 320 --preset mul_vegetation --augs True \
	--workers 6 --epochs 100 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber ln34_mul_vegetation_aug_dice\

python3 train_satellites.py \
	--arch unet11 --batch-size 15 \
	--imsize 320 --preset mul_vegetation --augs True \
	--workers 6 --epochs 100 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber un11_mul_vegetation_aug_dice\

python3 train_satellites.py \
	--arch linknet34 --batch-size 40 \
	--imsize 320 --preset mul_urban --augs True \
	--workers 6 --epochs 100 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber ln34_mul_urban_aug_dice\

python3 train_satellites.py \
	--arch unet11 --batch-size 15 \
	--imsize 320 --preset mul_urban --augs True \
	--workers 6 --epochs 100 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber un11_mul_urban_aug_dice\	

python3 train_satellites.py \
	--arch linknet34 --batch-size 40 \
	--imsize 320 --preset mul_blackwater --augs True \
	--workers 6 --epochs 100 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber ln34_mul_blackwater_aug_dice\

python3 train_satellites.py \
	--arch unet11 --batch-size 15 \
	--imsize 320 --preset mul_blackwater --augs True \
	--workers 6 --epochs 100 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber un11_mul_blackwater_aug_dice\

python3 train_satellites.py \
	--arch linknet34 --batch-size 40 \
	--imsize 320 --preset mul_ir1 --augs True \
	--workers 6 --epochs 100 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber ln34_mul_ir1_aug_dice\

python3 train_satellites.py \
	--arch unet11 --batch-size 15 \
	--imsize 320 --preset mul_ir1 --augs True \
	--workers 6 --epochs 100 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber un11_mul_ir1_aug_dice\	

python3 train_satellites.py \
	--arch linknet34 --batch-size 40 \
	--imsize 320 --preset mul_ir2 --augs True \
	--workers 6 --epochs 100 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber ln34_mul_ir2_aug_dice\

python3 train_satellites.py \
	--arch unet11 --batch-size 15 \
	--imsize 320 --preset mul_ir2 --augs True \
	--workers 6 --epochs 100 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber un11_mul_ir2_aug_dice\
