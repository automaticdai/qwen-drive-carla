"""Explicit full-precision and low-memory loading profiles for Qwen-Drive."""
from pathlib import Path


def load_model(model_path, *, precision="bf16", attention="sdpa", planner="sft"):
    import torch
    from qwen_drive import QwenDriveForPlanning
    if precision not in ("bf16", "nf4"):
        raise ValueError("precision must be bf16 or nf4")
    if planner not in ("sft", "rl"):
        raise ValueError("planner must be sft or rl")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen planning requires CUDA")
    model_path = Path(model_path)
    kwargs = dict(dtype=torch.bfloat16, attn_implementation=attention, local_files_only=True)
    if precision == "nf4":
        from transformers import BitsAndBytesConfig
        # The official planner is loaded separately after the VLM. Quantizing its
        # initially empty Linear modules would break that loader. Keep vision and
        # the tied output/embedding weights in BF16 too.
        kwargs.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["planning_expert", "vlm.model.visual", "vlm.lm_head"],
        ))
    model = QwenDriveForPlanning.from_pretrained(
        str(model_path), planner=str(model_path / f"planner-{planner}"), **kwargs)
    quantized = []
    if precision == "bf16":
        model = model.to("cuda")
    else:
        from bitsandbytes.nn import Linear4bit
        quantized = [name for name, module in model.named_modules() if isinstance(module, Linear4bit)]
        if not quantized or any(not name.startswith("vlm.model.language_model.") for name in quantized):
            raise RuntimeError("Unexpected NF4 module coverage; only language-model layers should be quantized")
        if any(p.dtype != torch.bfloat16 for p in model.planning_expert.parameters()):
            raise RuntimeError("The planning expert must remain in BF16")
        if any(p.dtype != torch.bfloat16 for p in model.vlm.model.visual.parameters()):
            raise RuntimeError("The vision module must remain in BF16")
    model.drive_loading_info = dict(precision=precision, planner=planner, quantized_linear_modules=len(quantized),
                                   vision_dtype="bfloat16", planner_dtype="bfloat16")
    return model.eval()
