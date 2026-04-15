import os
import importlib
from typing import Optional


DEFAULT_MODEL_ROOT = os.getenv("TRIDECODE_MODEL_ROOT", "/pscratch/sd/h/hmuki/models")
DEFAULT_LOCAL_FILES_ONLY = os.getenv("TRIDECODE_LOCAL_FILES_ONLY", "1").lower() not in {
    "0",
    "false",
    "no",
}


def resolve_model_path(model_name: str, model_root: Optional[str] = None) -> str:
    if os.path.isdir(model_name):
        return model_name

    root = model_root or DEFAULT_MODEL_ROOT
    candidate = os.path.join(root, model_name)
    if os.path.isdir(candidate):
        return candidate

    return model_name


def load_tokenizer(
    model_name: str,
    *,
    model_root: Optional[str] = None,
    local_files_only: Optional[bool] = None,
    **kwargs,
):
    transformers = importlib.import_module("transformers")
    AutoTokenizer = transformers.AutoTokenizer
    resolved_model_name = resolve_model_path(model_name, model_root=model_root)
    return AutoTokenizer.from_pretrained(
        resolved_model_name,
        local_files_only=DEFAULT_LOCAL_FILES_ONLY if local_files_only is None else local_files_only,
        **kwargs,
    )


def load_causal_lm(
    model_name: str,
    *,
    model_root: Optional[str] = None,
    local_files_only: Optional[bool] = None,
    **kwargs,
):
    transformers = importlib.import_module("transformers")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    resolved_model_name = resolve_model_path(model_name, model_root=model_root)
    return AutoModelForCausalLM.from_pretrained(
        resolved_model_name,
        local_files_only=DEFAULT_LOCAL_FILES_ONLY if local_files_only is None else local_files_only,
        **kwargs,
    )