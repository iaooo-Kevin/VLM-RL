from dataset_utils.add_cot_preprocess import makeTaggedMessages
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import SFTTrainer
from peft import LoraConfig


def messagesToText(example, tokeniser):
    messages = example["messages_tagged"]
    text = tokeniser.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt = False
    )
    return {"text": text}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseModel", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--datasetPath", type=str, default="AI-MO/NuminaMath-TIR")
    parser.add_argument("--outputDir", type=str, default="cot-qwen25-7b-10k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainSamples", type=int, default=10_000)
    parser.add_argument("--perDeviceBatch", type=int, default=2)
    parser.add_argument("--gradAccum", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--skipTagging", action="store_true", help="Assume messages_tagged already exists.")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 if your GPU supports it.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 if preferred.")
    args = parser.parse_args()

    #1) Load
    ds = load_dataset(args.datasetPath)

    #2) Inject tags (unless skipped)
    if not args.skipTagging:
        ds = ds.map(makeTaggedMessages, desc="Injecting <think>/<answer> tags")

    #3) Sample 10k from train split deterministically
    train = ds["train"].shuffle(seed=args.seed).select(range(min(args.trainSamples, len(ds["train"]))))

    #4) Tokeniser & model
    tokenizer = AutoTokenizer.from_pretrained(args.baseModel, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.baseModel,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.config.use_cache = False  # basbetter for training

    #
    trainText = train.map(lambda ex: messagesToText(ex, tokeniser=tokenizer),
                          remove_columns=train.column_names,
                          desc="Applying chat template")

    # 6) LoRA config (safe defaults for Qwen2.5 7B)
    loraCfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
    )

    # 7) Training args
    argsTrain = TrainingArguments(
        output_dir=args.outputDir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.perDeviceBatch,
        gradient_accumulation_steps=args.gradAccum,
        logging_steps=20,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        report_to="none",
        bf16=args.bf16,
        fp16=args.fp16,
        optim="adamw_torch",
        dataloader_num_workers=4
    )

    # 8) SFTTrainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=trainText,
        peft_config=loraCfg,
        # dataset_text_field="text",
        # packing=True,  #packs multiple samples into a long sequence for throughput
        args=argsTrain,
    )

    trainer.train()

    # Save adapters + tokenizer
    trainer.model.save_pretrained(args.outputDir)
    tokenizer.save_pretrained(args.outputDir)

    # If you want a **merged** FP16 model for easy deployment, uncomment:
    # from peft import AutoPeftModelForCausalLM
    # merged = AutoPeftModelForCausalLM.from_pretrained(args.outputDir, device_map="auto")
    # merged = merged.merge_and_unload()
    # merged.save_pretrained(os.path.join(args.outputDir, "merged"), safe_serialization=True)
    # tokenizer.save_pretrained(os.path.join(args.outputDir, "merged"))

    print("Done. Model saved to:", args.outputDir)

if __name__ == "__main__":
    # main()
    from eval.numina_math import math_eval
    math_eval.main()

