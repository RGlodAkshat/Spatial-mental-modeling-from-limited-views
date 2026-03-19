import json
import re


def extract_json_from_text(text: str):
    if not text:
        return None
    code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    json_matches = re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    for match in json_matches:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()


def normalize_direction(direction: str) -> str:
    if not direction:
        return ""
    direction = direction.lower().strip()
    mapping = {
        'north': 'up', 'south': 'down', 'west': 'left', 'east': 'right',
        'forward': 'inner', 'backward': 'outer', 'front': 'inner', 'back': 'outer'
    }
    return mapping.get(direction, direction)
