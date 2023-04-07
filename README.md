# Fine-Tuning a Stable Diffusion Model with HuggingFace's Diffusers Library

In this tutorial, we'll walk through the process of fine-tuning a stable diffusion model using HuggingFace's diffusers library. The stable diffusion model is a generative model that can be used for a variety of tasks, including image synthesis, text generation, and audio generation.

## Hardware

Running Stable Diffusion itself is not too demanding by today's standards, and fine tuning the model doesn't require anything like the hardware on which it was originally trained.

## Data

[BLIP Flowers Dataset on Hugging Face](https://huggingface.co/datasets/pranked03/flowers-blip-captions)

If you want to create your own dataset containing **text-image pairs**, this [Github Repository](https://github.com/pranavgupta2603/BLIP-flower-captioning) of mine will help you out.

## Fine Tuning

Use the [Python Notebook](https://github.com/pranavgupta2603/flowers-sd-finetuning/blob/main/flowers_sd_finetune.ipynb) in the repository
---
The exclamation mark (!) to run shell commands or terminal commands directly from within a code cell. When you use the exclamation mark before a command, Jupyter interprets it as a shell command rather than a Python statement.

The code below is present in the Notebook.
```Python
!accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --dataset_name="HUGGINGFACE_DATASET_NAME" \
  --use_ema \
  --resolution=128 --center_crop --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=1000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="PATH_TO_SAVE_MODEL" 
```
Change the ```--dataset_name="HUGGINGFACE_DATASET_NAME"``` to a dataset containing image pairs on HuggingFace
