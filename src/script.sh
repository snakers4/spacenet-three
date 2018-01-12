python3 train_satellites.py \
	--arch unet11 --batch-size 15 \
	--imsize 320 --preset mul_ir2 --augs True \
	--workers 6 --epochs 50 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber un11_mul_ir2_aug_dice\	
