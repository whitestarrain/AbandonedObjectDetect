# python prune_finetune.py --img 640 --batch 16 --epochs 100 --data data/coco4.yaml --yaml-cfg modules/yolov5s.yaml  --cfg ./cfg/yolov5s_v6_coco4.cfg --weights ./last.pt --name distill_train --distill

# python prune_finetune.py --img 640 --batch 16 --epochs 10 --data data/coco-filter.yaml --cfg ./cfg/prune_0.8_keep_0.01_yolov5s_v6_coco-filter.cfg --weights ./runs/prune_0.8_keep_0.01_train/prune_0.8_keep_0.01_sparsity_coco2/prune_0.8_keep_0.01_weights/prune_0.8_keep_0.01_last.pt --t_weight runs/train/exp4/weights/last.pt  --name s_coco_finetune_distill --distill

python prune_finetune.py --img 640 --batch 16 --epochs 50 --data data/coco-filter.yaml --cfg ./cfg/prune_0.8_keep_0.01_yolov5s_v6_coco-filter.cfg --weights runs/prune_0.8_keep_0.01_train/prune_0.8_keep_0.01_sparsity_coco-filter/prune_0.8_keep_0.01_weights/prune_0.8_keep_0.01_best.pt --t_weight runs/train/coco-filter_epoch50/weights/best.pt  --name s_coco-filter_finetune_distill 
# --distill

