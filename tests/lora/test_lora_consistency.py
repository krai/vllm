import json
import os
import tempfile
from typing import List, Dict

import pytest
import vllm
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from huggingface_hub import snapshot_download

# Initialize LLM engine for Llama3.1-8B
def get_llm(enable_lora: bool) -> vllm.LLM:
    return vllm.LLM(
        model="meta-llama/Llama-3.1-8b",
        device="cuda",  
        enable_lora=enable_lora,
        max_model_len=4096,  
        max_num_seqs=2,     
        max_num_batched_tokens=4096,  
        swap_space=32,
        max_lora_rank=32
    )

def format_output(outputs: List[RequestOutput], start_idx: int = 0) -> List[Dict]:
    return [
        {
            "qsl_idx": i + start_idx,
            "seq_id": i + start_idx,
            "data": output.outputs[0].text
        }
        for i, output in enumerate(outputs)
    ]

@pytest.fixture
def sample_prompts():
    return [
        "Describe the theory of evolution.",
        "Explain quantum mechanics in simple terms.",
        "Write a short story about a time traveler.",
        "What are the main principles of democracy?",
    ]

@pytest.fixture
def llama_adapters():
    adapter_1 = snapshot_download(repo_id="pbevan11/llama-3.1-8b-ocr-correction")
    adapter_2 = snapshot_download(repo_id="RikiyaT/Meta-Llama-3.1-8B-LoRA-test")
    return [("adapter_llama_1", adapter_1, 1), ("adapter_llama_2", adapter_2, 2)]

@pytest.fixture
def sampling_params():
    return vllm.SamplingParams(max_tokens=64, temperature=0.7)

@pytest.fixture
def output_dir():
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Test: Add Adapter
def test_add_llama_adapter(sample_prompts, llama_adapters, sampling_params, output_dir):
    llm = get_llm(enable_lora=True)

    # Add and generate with one adapter
    adapter_name, adapter_path, adapter_id = llama_adapters[0]
    print(f"\nAdding adapter: {adapter_name}")

    lora_request = LoRARequest(
        lora_name=adapter_name,
        lora_int_id=adapter_id,
        lora_path=adapter_path,
    )
    llm.llm_engine.add_lora(lora_request)

    outputs = llm.generate(sample_prompts, sampling_params)
    formatted_outputs = format_output(outputs)
    assert len(formatted_outputs) == len(sample_prompts), "Output generation failed for add_adapter"

    # Save outputs
    output_file = os.path.join(output_dir, f"{adapter_name}_add_adapter_outputs.json")
    with open(output_file, "w") as f:
        json.dump(formatted_outputs, f, indent=2)

# Test: Remove Adapter
def test_remove_llama_adapter(sample_prompts, llama_adapters, sampling_params):
    llm = get_llm(enable_lora=True)

    # Add an adapter
    adapter_name, adapter_path, adapter_id = llama_adapters[0]
    llm.llm_engine.add_lora(LoRARequest(lora_name=adapter_name, lora_int_id=adapter_id, lora_path=adapter_path))

    # Remove the adapter
    llm.llm_engine.remove_lora(adapter_id)

    # Generate output 
    outputs = llm.generate(sample_prompts, sampling_params)
    assert len(outputs) == len(sample_prompts), "Output generation failed after removing adapter"

# Test: Swap between Adapters
def test_swap_llama_adapters(sample_prompts, llama_adapters, sampling_params, output_dir):
    llm = get_llm(enable_lora=True)

    # Add the first adapter
    adapter_1_name, adapter_1_path, adapter_1_id = llama_adapters[0]
    llm.llm_engine.add_lora(
        LoRARequest(lora_name=adapter_1_name, lora_int_id=adapter_1_id, lora_path=adapter_1_path)
    )

    # Generate output with the first adapter
    outputs_adapter_1 = llm.generate(sample_prompts, sampling_params)
    formatted_outputs_1 = format_output(outputs_adapter_1)

    # Simulate swap by removing the first adapter and adding the second adapter
    llm.llm_engine.remove_lora(adapter_1_id)
    
    adapter_2_name, adapter_2_path, adapter_2_id = llama_adapters[1]
    llm.llm_engine.add_lora(
        LoRARequest(lora_name=adapter_2_name, lora_int_id=adapter_2_id, lora_path=adapter_2_path)
    )

    # Generate output with the second adapter
    outputs_adapter_2 = llm.generate(sample_prompts, sampling_params)
    formatted_outputs_2 = format_output(outputs_adapter_2)

    with open(os.path.join(output_dir, "swap_adapter_1_outputs.json"), "w") as f:
        json.dump(formatted_outputs_1, f, indent=2)
    with open(os.path.join(output_dir, "swap_adapter_2_outputs.json"), "w") as f:
        json.dump(formatted_outputs_2, f, indent=2)

    # Ensure outputs from both adapters are different
    assert formatted_outputs_1 != formatted_outputs_2, "Swapping adapters did not produce different outputs"
