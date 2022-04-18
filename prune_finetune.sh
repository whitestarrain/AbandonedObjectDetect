python prune_finetune.py --img 640 --batch 16 --epochs 100 --data data/coco4.yaml --yaml-cfg modules/yolov5s.yaml  --cfg ./cfg/yolov5s_v6_coco4.cfg --weights ./last.pt --name distill_train --distill
