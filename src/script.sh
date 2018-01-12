python3 train_satellites.py \
	--arch linknet34 --batch-size 40 \
	--imsize 320 --preset mul_8channel --augs True \
	--workers 6 --epochs 100 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber ln34_mul_8channel_aug_dice\

