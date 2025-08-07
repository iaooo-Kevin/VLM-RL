from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModelForImageTextToText
from peft import PeftModel, PeftConfig
import torch
import json
import re
import logging
from qwen_vl_utils import process_vision_info
from math_verify import parse, verify
from tqdm import tqdm
import argparse

#Setup logging
logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--model_path_or_id", type=str, required=True, help='Path to the base model or ID as seen on Huggingface')
    arguments.add_argument("--adapter_path", type=str, required=False, help='Path to the model adapter. Provide this if you trained a LoRA model')
    arguments.add_argument("--batch_size", type=int, default=50, help="Batch-size while running inference. Defaults to 50. Try reducing in case of OOM")
    arguments.add_argument("--output_path", type=str, default='/mnt/disk1/grpo_vlm/eval/geometry/logs/GeoQA.json', help='Path to store the results from running the evaluation')
    args = arguments.parse_args()

    if args.adapter_path is not None:
        logger.info("Merging base model and lora adaptors..")
        baseModel = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path_or_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(
            args.model_path_or_id,
            use_fast=True
        )
        processor.tokenizer.padding_side='left'
        model = PeftModel.from_pretrained(
            baseModel,
            args.adapter_path
        )
    else:
        logger.info("Just using the base model for inference.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path_or_id,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        processor = AutoProcessor.from_pretrained(
            args.model_path_or_id,
            use_fast=True,
        )
        processor.tokenizer.padding_side='left'
    
    promptPath = "./prompts/geoqa_test_prompts.jsonl"

    data = []
    with open(promptPath, "r") as evalfile:
        for line in evalfile:
            data.append(json.loads(line))

    QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    messages = []

    data = data
    for i in data:
        message = [{
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": f"file://{i['image_path']}"
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=i['question'])
            }
        ]
        }]
        messages.append(message)
    allOutputs = []

    for i in tqdm(range(0, len(messages), args.batch_size)):
        batchMessages = messages[i : i + args.batch_size]
        text = [processor.apply_chat_template(msg) for msg in batchMessages]
        imageInputs, videoInputs = process_vision_info(batchMessages)
        inputs = processor(
            text=text,
            images=imageInputs,
            videos=videoInputs,
            padding=True,
            return_tensors='pt'
        )
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generatedIDs = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=1024,
                do_sample=False
            )
        generatedTrimmed = [
            outIds[len(inpIds): ] for inpIds, outIds in zip(inputs["input_ids"], generatedIDs)
        ]
        batchOutput = processor.batch_decode(
            generatedTrimmed,
            skip_special_tokens=True,
            cleanup_tokenization_spaces=False
        )
        allOutputs.extend(batchOutput)
        print(f"Processed batch {i//args.batch_size + 1}/{(len(messages) + args.batch_size - 1)//args.batch_size}")
    
    #Now, then,
    finalOutput, correctNumber = [], 0
    for inputExample, modelOutput in zip(data, allOutputs):
        original = modelOutput
        groundTruth = inputExample["ground_truth"]
        modelAnswer = parse(original)
        if modelAnswer is not None and float(verify(modelAnswer, parse(groundTruth))) > 0:
            correctNumber += 1
            isCorrect = True
        else:
            isCorrect = False
        try:
            result = {
                "question": inputExample,
                "ground_truth": groundTruth,
                "model_answer": original,
                "extracted_answer": str(modelAnswer[0]) if modelAnswer else None,
                "isCorrect": isCorrect
            }
        except Exception as e:
            print("No answer Parsed", e, modelAnswer)
            result = {
                "question": inputExample,
                "ground_truth": groundTruth,
                "model_answer": modelAnswer,
                "extracted_answer": None,
                "isCorrect": isCorrect
            }
        finalOutput.append(result)
    
    #Calculate Accuracy and print.
    Accuracy = correctNumber / len(data) * 100
    print(f'Accuracy: {Accuracy:.2f}%')

    #Save results to a JSON file.
    outputPath = args.output_path
    with open(outputPath, "w") as f:
        json.dump({
            "accuracy": Accuracy,
            "results": finalOutput
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to output: {outputPath}")


if __name__ == "__main__":
    main()