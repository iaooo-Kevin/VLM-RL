from datasets import load_dataset, Dataset, Image
from typing import Optional, Tuple
from PIL import Image as PIL_Image
import io


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

def process_example(example):
    image_list = example["images"]
    if isinstance(image_list, PIL_Image.Image):
        example['image'] = image_list.convert("RGB") if image_list.mode != "RGB" else image_list
    elif isinstance(image_list, list) and len(image_list) == 1:
        pil_image = image_list[0]
        if isinstance(pil_image, PIL_Image.Image):
            if pil_image.mode != "RGB":
                example['image'] = pil_image.convert("RGB")
            else:
                example['image'] = pil_image
        else:
            example['image'] = PIL_Image.open(io.BytesIO(pil_image['bytes'])).convert("RGB")
    else:
        example['image'] = None
    example['solution'] = example['answer']
    example['prompt'] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
            ],
        },
    ]
    return example



def load_processed_dataset(
    dataset_name: str,
    test_size: Optional[int] = None,
    seed: int = 42,
    train_split: str = "train"
) -> Tuple[Dataset, Optional[Dataset]]:
    raw_dataset = load_dataset(dataset_name, split=train_split)

    if test_size:
        dataset_dict = raw_dataset.train_test_split(test_size=test_size, seed=seed)
    else:
        dataset_dict = {"train": raw_dataset}

    for split in dataset_dict:
        dataset_dict[split] = dataset_dict[split].map(
            process_example,
            remove_columns=['images', 'answer']
        )
    return dataset_dict["train"], dataset_dict.get("test")

if __name__ == "__main__":
    datasetID = 'DaveKevin/GeoQA-GoldenCoT' #DaveKevin/GeoQA-GoldenCoT, #lmms-lab/multimodal-open-r1-8k-verified
    trainDataset, evalDataset = load_processed_dataset(dataset_name=datasetID)
#    print(f"train dataset: {trainDataset}")
    print(f'First row of the trainDataset: {trainDataset[0]}')
    #    print(f"Eval Dataset: {evalDataset}")