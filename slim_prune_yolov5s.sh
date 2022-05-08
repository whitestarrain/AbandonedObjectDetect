python slim_prune_yolov5s.py  --cfg cfg/yolov5s_v6_coco-filter.cfg  --data data/prune/coco-filter.data --weights runs/train/sparsity_coco-filter/weights/best.pt   --global_percent 0.8 --layer_keep 0.01

# mkdir runs/prune_0.8_keep_0.01_train/prune_0.8_keep_0.01_sparsity_coco2/prune_0.8_keep_0.01_weights/prune_0.8_keep_0.01_last.pt
# mkdir runs/prune_0.8_keep_0.01_train/prune_0.8_keep_0.01_sparsity_coco-filter/prune_0.8_keep_0.01_weights

