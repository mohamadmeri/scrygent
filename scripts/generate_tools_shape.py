import json

from scrygent.models.registry import TOOL_PARAM_MODELS

for tool, model in TOOL_PARAM_MODELS.items():
    print("=" * 80)
    print(tool.value)
    print(json.dumps(model.model_json_schema(), indent=2))