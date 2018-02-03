python3 train_gapnet.py \
	--arch gapnet18 --batch-size 4 --dilation 40 \
	--imsize 640 --preset mul_ps_vegetation --augs True \
	--workers 6 --epochs 50 --start-epoch 0 \
	--seed 42 --print-freq 20 \
	--lr 1e-4 --optimizer adam \
	--tensorboard True --tensorboard_images True \
	--lognumber gapnet_img18_vegetation_dilation_40
