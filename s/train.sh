#! /bin/bash
python train_controlnet3d.py \
  --use_8bit_adam \
  --gradient_checkpointing \
  --gradient_accumulation_steps 8 \
  --train_batch_size 1 \
  --seed 42 \
  --output_dir output \
  --pretrained_model_name_or_path /home/jonathan/models/zeroscope_v2_576w \
  --train_data_dir caption_output_test_small \
  --mixed_precision bf16 