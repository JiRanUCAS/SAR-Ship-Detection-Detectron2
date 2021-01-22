############################## HRSID ###################################
python train_retinanet.py \
  --config-file yamls/HRSID/Retinanet-R50.yaml  \
  --num-gpus 2 \
  --dist-url "tcp://127.0.0.1:50001"
  
    
python train_fasterrcnn.py \
  --config-file yamls/HRSID/Faster-RCNN-R50.yaml  \
  --num-gpus 2 \
  --dist-url "tcp://127.0.0.1:50001"
  
############################## SSDD ###################################
python train_retinanet.py \
  --config-file yamls/SSDD/Retinanet-R50.yaml  \
  --num-gpus 2 \
  --dist-url "tcp://127.0.0.1:50001"

python train_fasterrcnn.py \
  --config-file yamls/SSDD/Faster-RCNN-R50.yaml  \
  --num-gpus 2 \
  --dist-url "tcp://127.0.0.1:50001"
  
############################## Ship_Part1 ###################################
python train_retinanet.py \
  --config-file yamls/Ship_Part1/Retinanet-R50.yaml  \
  --num-gpus 2 \
  --dist-url "tcp://127.0.0.1:50001"

python train_fasterrcnn.py \
  --config-file yamls/Ship_Part1/Faster-RCNN-R50.yaml  \
  --num-gpus 2 \
  --dist-url "tcp://127.0.0.1:50001"
  
############################## Ship ###################################
python train_retinanet.py \
  --config-file yamls/Ship/Retinanet-R50.yaml  \
  --num-gpus 2 \
  --dist-url "tcp://127.0.0.1:50001"

python train_fasterrcnn.py \
  --config-file yamls/Ship/Faster-RCNN-R50.yaml  \
  --num-gpus 2 \
  --dist-url "tcp://127.0.0.1:50001"  