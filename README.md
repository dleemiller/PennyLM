# PennyLM
Training scripts for Penny-1.7B

# Step 1 (Optional) - Create Dataset

Use the jupyter notebook `create_ipj_dataset.ipynb` to create the dataset from scratch.

Or use the parquet file from (huggingface)[https://huggingface.co/datasets/dleemiller/irish_penny_journal]


# Step 2 - Train Classifier

In my script, I used MiniLMv2 6-layer 384-hidden dim. This is a very small BERT-like transformer model (22MB), so it doesn't take a lot of resources and runs quickly.

Since the classification task for this dataset is pretty easy to learn, might as well use a small model that you could run on CPU if you wanted.

If you were to change to a more difficult classification task, then you might want to use a different model.


# Step 3 - GRPO Training

I was able to load the classifier and SmolLM2-1.7B for full fine-tuning even with a setting of 12 generations per step. This fit in 48GB of VRAM. If you have more or less to work with, consider LoRA or unsloth, or fewer generations (probably not less than 6).
