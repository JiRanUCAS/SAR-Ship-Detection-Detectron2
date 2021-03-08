############################## HRSID ###################################
# python train_fasterrcnn.py \
#   --config-file yamls/HRSID/Faster-RCNN-R50.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"  

# python train_retinanet.py \
#   --config-file yamls/HRSID/Faster-RCNN-R18.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"
  
# python train_cascade_mask_rcnn.py \
#   --config-file yamls/HRSID/Cascade-MaskRCNN-R50.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001" 
  
CUDA_VISIBLE_DEVICES=0,1 python train_mask_rcnn.py \
  --config-file yamls/HRSID/MaskRCNN-R50.yaml  \
  --num-gpus 2 \
  --dist-url "tcp://127.0.0.1:50001" 
  
# python train_retinanet.py \
#   --config-file yamls/HRSID/Retinanet-R18.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001" 
  
# python train_retinanet.py \
#   --config-file yamls/HRSID/Retinanet-R50.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"

  

    
############################## SSDD ###################################
# python train_cascade_mask_rcnn.py \
#   --config-file yamls/SSDD/Cascade-MaskRCNN-R50.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001" 
  
# python train_retinanet.py \
#   --config-file yamls/SSDD/Retinanet-R18.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"
  
# python train_retinanet.py \
#   --config-file yamls/SSDD/Retinanet-R50.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"

# python train_fasterrcnn.py \
#   --config-file yamls/SSDD/Faster-RCNN-R18.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"
  
# python train_fasterrcnn.py \
#   --config-file yamls/SSDD/Faster-RCNN-R50.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"

############################## Ship_Part1 ###################################
# python train_retinanet.py \
#   --config-file yamls/Ship_Part1/Retinanet-R18.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"

# python train_retinanet.py \
#   --config-file yamls/Ship_Part1/Retinanet-R50.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"
  
# python train_fasterrcnn.py \
#   --config-file yamls/Ship_Part1/Faster-RCNN-R18.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"

# python train_fasterrcnn.py \
#   --config-file yamls/Ship_Part1/Faster-RCNN-R50.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"

############################## Ship ###################################
# python train_retinanet.py \
#   --config-file yamls/Ship/Retinanet-R18.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"
  
# python train_retinanet.py \
#   --config-file yamls/Ship/Retinanet-R50.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"

  
# python train_fasterrcnn.py \
#   --config-file yamls/Ship/Faster-RCNN-R18.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"  

# python train_fasterrcnn.py \
#   --config-file yamls/Ship/Faster-RCNN-R50.yaml  \
#   --num-gpus 2 \
#   --dist-url "tcp://127.0.0.1:50001"    