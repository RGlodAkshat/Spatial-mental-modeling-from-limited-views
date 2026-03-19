"""
Data Formatters for Model Training
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Literal
from ..utils import load_jsonl, save_json, ensure_dir

ModelType = Literal["qwen2.5vl", "llava", "instructblip"]


class ModelDataFormatter(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def format_conversation(self, item: Dict) -> Dict:
        pass

    @abstractmethod
    def get_output_filename(self, input_filename: str) -> str:
        pass

    def validate_item(self, item: Dict) -> bool:
        required_fields = ["input_prompt", "grounded_output", "images"]
        return all(field in item for field in required_fields) and isinstance(item["images"], list)

    def convert_data(self, prompt_data: List[Dict]) -> List[Dict]:
        converted_data = []
        for item in prompt_data:
            if self.validate_item(item):
                converted_data.append(self.format_conversation(item))
        return converted_data


class QwenDataFormatter(ModelDataFormatter):
    def __init__(self):
        super().__init__("qwen2.5vl")

    def format_conversation(self, item: Dict) -> Dict:
        images = item["images"]
        input_prompt = item["input_prompt"]
        grounded_output = item["grounded_output"]
        image_placeholders = "\n".join(["<image>" for _ in images])
        human_value = f"{image_placeholders}\n{input_prompt}"
        return {
            "images": images,
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": grounded_output},
            ],
        }

    def get_output_filename(self, input_filename: str) -> str:
        base_name = os.path.splitext(input_filename)[0]
        return f"{base_name}_qwen_sft.json"


FORMATTER_REGISTRY = {"qwen2.5vl": QwenDataFormatter()}


def get_formatter(model_type: ModelType) -> ModelDataFormatter:
    return FORMATTER_REGISTRY[model_type]


def list_supported_models() -> List[str]:
    return list(FORMATTER_REGISTRY.keys())


def convert_prompts_to_sft_format(input_file: str, output_file: str, model_type: ModelType) -> None:
    prompt_data = load_jsonl(input_file)
    formatter = get_formatter(model_type)
    converted_data = formatter.convert_data(prompt_data)
    ensure_dir(os.path.dirname(output_file))
    save_json(converted_data, output_file)


def batch_convert_prompts_to_sft(input_dir: str, output_dir: str, model_type: ModelType) -> None:
    import glob
    ensure_dir(output_dir)
    formatter = get_formatter(model_type)
    for input_file in glob.glob(os.path.join(input_dir, "*.jsonl")):
        output_filename = formatter.get_output_filename(os.path.basename(input_file))
        convert_prompts_to_sft_format(input_file, os.path.join(output_dir, output_filename), model_type)
