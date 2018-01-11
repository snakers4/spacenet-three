python3 train_satellites.py \
	--arch linknet34 --batch-size 40 \
	--imsize 320 --preset mul_ir1 --augs False \
	--workers 6 --epochs 50 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber ln34_mulir1_noaug_dice\

python3 train_satellites.py \
	--arch unet11 --batch-size 20 \
	--imsize 320 --preset mul_ir1 --augs False \
	--workers 6 --epochs 50 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-3 --optimizer adam \
	--tensorboard True --tensorboard_images True --lognumber un11_mulir1_noaug_dice\
