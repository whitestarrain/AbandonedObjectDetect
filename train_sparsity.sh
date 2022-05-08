python train_sparsity.py --img 640 --batch 16 --epochs 50 --data data/coco-filter.yaml --cfg models/yolov5s.yaml --weights runs/train/coco-filter_epoch50/weights/best.pt  --name sparsity_coco-filter -sr --scale 0.001 --prune 1

