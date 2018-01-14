python3 train_satellites.py \
	--arch unet11 --batch-size 15 \
	--imsize 320 --preset mul_naive1 --augs True \
	--workers 6 --epochs 100 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber un11_mul_naive1_aug_dice\
