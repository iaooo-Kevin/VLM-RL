import torch
from dataclasses import dataclass, field
from trl import (
    GRPOTrainer,
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
    get_kbit_device_map,
    get_quantization_config,
)

from dataset_utils.preprocess import load_processed_dataset
from rewards.accuracy_reward import accuracy_reward
from trl.rewards import think_format_reward
from rewards.reasoning_reward import reasoningReward
@dataclass
class CustomScriptArgs(ScriptArguments):
    use_gspo: bool = field(default=False, metadata={"help": "Whether or not to use GSPO. "})
    intermediate_reasoning: bool = field(default=False, metadata={"help": "Include intermediate reasoning rewards"})
    use_dapo: bool = field(default=False, metadata={"help": "Whether or not to use DAPO."})
    use_drgrpo: bool = field(default=False, metadata={"help": "Whether or not to use DAPO."})

def main():
    parser = TrlParser((CustomScriptArgs, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    losses = sum([
        script_args.use_gspo, script_args.use_dapo, script_args.use_drgrpo
    ])
    if losses > 1:
        raise ValueError(
            "Only one of `--use_gspo`, `--use_dapo`, or `--use_drgrpo` can be set to True."
        )

    if script_args.use_gspo:
        print(f'Applying GSPO loss.')
        training_args.importance_level_sampling = 'sequence'
        training_args.loss_type = 'grpo'
        training_args.beta = 0.0
        training_args.epsilon = 4e-4
        training_args.steps_per_generation = 4 #Must be 4x the gradient accumulation steps.
        training_args.gradient_accumulation_steps = 1
    if script_args.use_dapo:
        print(f'Applying DAPO loss.')
        training_args.loss_type='dapo'
        training_args.epsilon=0.2
        training_args.epsilon_high=0.28
    if script_args.use_drgrpo:
        print(f'Appling Dr. GRPO loss.')
        training_args.loss_type='dr_grpo'
        training_args.mask_truncated_completions=True
        training_args.epsilon=0.2
        training_args.epsilon_high=0.28
        training_args.beta = 0.0

    #Handle quant configuration.
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quant_config = get_quantization_config(model_args)

    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quant_config else None,
        quantization_config=quant_config,
    )

    #Load your dataset.
    train_dataset, eval_dataset = load_processed_dataset(
        dataset_name=script_args.dataset_name,
        test_size=100,
        seed=42,
        train_split=script_args.dataset_train_split,
    )

    #Reward functions
    reward_funcs = [think_format_reward, accuracy_reward]
    if script_args.intermediate_reasoning:
        reward_funcs.append(reasoningReward)
    
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    # Train and save
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    main()