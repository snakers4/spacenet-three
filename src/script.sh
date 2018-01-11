python3 train_satellites.py \
		--arch linknet34 --batch-size 40 \
		--imsize 320 --preset mul_ir1 --augs False \
		--workers 6 --epochs 30 --start-epoch 0 \
		--seed 42 --print-freq 10 \
		--lr 1e-2 --optimizer adam \
		--tensorboard True --tensorboard_images True --lognumber test2\
