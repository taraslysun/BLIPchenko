# finetune_poetry.py
import argparse
import wandb
from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from peft import  get_peft_model, LoraConfig
from datasets import load_dataset, Dataset
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

def get_dataset(args):
    if args.data_format == "raw":
        # raw continuous text loader (as above)
        lines = [l.strip() for l in open(args.data_path, encoding="utf-8") if l.strip()]
        chunks = ["\n".join(lines[i:i+args.chunk_size]) for i in range(0,len(lines),args.chunk_size)]
        return Dataset.from_dict({"text": chunks})
    elif args.data_format == "jsonl":
        return load_dataset("json", data_files=args.data_path, split="")
    else:
        raise ValueError("Unknown format")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_format", choices=["raw","jsonl"], required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk_size", type=int, default=20)
    args = parser.parse_args()

    ds = get_dataset(args)

    # 2) Load model + tokenizer via Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/Llama-2-7b-hf", device_map="auto" 
    )

    # 3) PEFT/LoRA config
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], 
        lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        # fallback hook registration
        def make_inputs_require_grad(module, inp, out):
            out.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # 4) Training arguments
    training_args = UnslothTrainingArguments(
        gradient_checkpointing=False,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        output_dir=args.output_dir,
        optim="paged_adamw_32bit",
        # fp16=True,
        bf16=True,
        # evaluation_strategy="no"
    )

    # 5) Initialize W&B
    wandb.init(project="ukr-poetry-llama2", config=vars(args))

    # 6) SFT Trainer
    trainer = UnslothTrainer(
        model=model,
        train_dataset=ds,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )

    # 7) Train!
    trainer.train()
    trainer.save_model()

    wandb.finish()

if __name__=="__main__":
    main()


# python finetune_poetry.py \
#   --data_path data/poems_all.txt \
#   --data_format raw \
#   --output_dir out_raw \
#   --epochs 3

# python finetune_poetry.py \
#   --data_path data/shevchenko_poems.txt \
#   --data_format raw \
#   --output_dir out_small \
#   --epochs 3

# python finetune_poetry.py \
#   --data_path data/taras_shevchenko_poetry.jsonl \
#   --data_format jsonl \
#   --output_dir out_jsonl \
#   --epochs 3
