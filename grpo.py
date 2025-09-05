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
import logging
from dataset_utils.preprocess import load_processed_dataset
from rewards.accuracy_reward import accuracy_reward
from rewards.accuracy_mcq_reward import mcq_reward
from trl.rewards import think_format_reward
from rewards.medical_reward import medical_rewards
from rewards.format_reward_forgeoqa import makeFormatReward
from rewards.reasoning_reward import reasoningReward, reasoning_reward

#Set up logging
logging.basicConfig(
    level=logging.INFO
)
#Set httpx from info to warning to avoid too much logging.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class CustomScriptArgs(ScriptArguments):
    use_gspo: bool = field(default=False, metadata={"help": "Whether or not to use GSPO. "})
    intermediate_reasoning: bool = field(default=False, metadata={"help": "Include intermediate reasoning rewards"})
    use_dapo: bool = field(default=False, metadata={"help": "Whether or not to use DAPO."})
    use_drgrpo: bool = field(default=False, metadata={"help": "Whether or not to use DAPO."})
    use_mcq_reward: bool = field(default=False, metadata={"help": "Whether or not to use MCQ reward function (string matching)."})
    use_llm_judge: bool = field(default=False, metadata={"help": "Whether or not to use LLM-as-a-judge for accuracy rewards."})
    use_medical_reward: bool = field(default = False, metadata = {"help": "Whether to enable medical rewards. Keep it disabled for non-medical datasets."})
    lora_rank: int = field(default = 16, metadata={"help": "The LoRA rank for the model, if you increase the rank, more number of model parameters will be trained."})

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
    if training_args.max_prompt_length == -1:
        training_args.max_prompt_length = None

    if script_args.use_gspo:
        logger.info(f'Applying GSPO loss.')
        training_args.importance_level_sampling = 'sequence'
        training_args.loss_type = 'grpo'
        training_args.beta = 0.0
        training_args.epsilon = 4e-4
        training_args.steps_per_generation = 4 #Must be 4x the gradient accumulation steps.
        training_args.gradient_accumulation_steps = 1
    if script_args.use_dapo:
        logger.info(f'Applying DAPO loss.')
        training_args.loss_type='dapo'
        training_args.epsilon=0.2
        training_args.epsilon_high=0.28
    if script_args.use_drgrpo:
        logger.info(f'Applying Dr. GRPO loss.')
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

    #Shuffle the training dataset for good measure,
    train_dataset = train_dataset.shuffle(seed=42)

    #Reward functions
    format_reward = makeFormatReward("answer")
    #Main Reward functions, the first three are accuracy rewards, a little bit about them and when to use which
    # 1). use_llm_judge: This uses an LLM to compare the model's answer with the ground truth; However this method though looks robust on the surface, introduces high variance owing to the inherent stochasticity of LLMs.
    # 2). use_mcq_reward: This is a simple string matching reward function for multiple choice questions. It extracts the answer from <answer> tags and compares it with the ground truth. This method is suitable for MCQs where answers are typically short and well-defined.
    # 3). Default (symbolic + string matching): This method first tries to use huggingface's `math_verify` to extract and symbolically compare the answers, if that fails it falls back to string matching. Suitable for mathematical questions.
    if script_args.use_llm_judge:
        logger.info(f'Using LLM-as-a-judge for accuracy rewards.')
        from rewards.acc_llm_reward import AccuracyLLM
        reward_funcs = [format_reward, AccuracyLLM]
    elif script_args.use_mcq_reward:
        logger.info(f'Using string matching for MCQ accuracy rewards.')
        reward_funcs = [format_reward, mcq_reward]
    else:
        logger.info(f'Using symbolic matching and string matching for accuracy rewards.')
        reward_funcs = [format_reward, accuracy_reward]
    if script_args.intermediate_reasoning:
        reward_funcs.append(reasoningReward)
    if script_args.use_medical_reward:
        logger.info(f'You have enabled medical rewards. Ensure that your dataset is medical in nature, else the rewards may not be meaningful.')
        reward_funcs.append(medical_rewards)

    if script_args.lora_rank != 16:
        logger.info(f'Using LoRA rank: {script_args.lora_rank}')
        model_args.lora_r = script_args.lora_rank
    else:
        logger.info(f'Using the default LoRA rank: {model_args.lora_r}')
        model_args.lora_r = 16 #Use the default value.

    logger.info(f'Peft configuration for the model: {get_peft_config(model_args)}')
    #Note: Disabling peft-config will use the full model. You can remove `--use_peft` from the command line to do a full RL training. Although, this is not recommended as this GPU memory is not sufficient for full model training.
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )
    trainableParameters = 0
    allParameters = 0
    for _, param in trainer.model.named_parameters():
        numParams = param.numel()
        allParameters += numParams
        if param.requires_grad:
            trainableParameters += numParams
        
    logger.info(f"Number of trainable parameters: {trainableParameters}")
    logger.info(f"As a percentage of total parameters: {100 * trainableParameters/allParameters:.2f}%")

    # Train and save
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    main()