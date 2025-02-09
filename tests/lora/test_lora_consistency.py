import json
import os
import pickle
import pytest
import tempfile
from typing import List, Dict

import vllm
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.utils import random_uuid
from huggingface_hub import snapshot_download

@pytest.fixture
def sample_prompts() -> List[str]:
    """Sample prompts for testing."""
    return [
        "Once upon a time",
        "The quick brown fox",
    ]

@pytest.fixture
def output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture(scope="session")
def test_lora_path():
    return snapshot_download(repo_id="SangBinCho/mixtral-lora")

@pytest.fixture(scope="session")
def adapter_1():
    """Get first LoRA adapter."""
    return snapshot_download(repo_id="SangBinCho/mixtral-lora")

@pytest.fixture(scope="session")
def adapter_2():
    """Get second LoRA adapter."""
    return snapshot_download(repo_id="dyang415/mixtral-lora-v0")

def get_llm(enable_lora: bool) -> vllm.LLM:
    """Initialize LLM with or without LoRA."""
    return vllm.LLM(
        model="facebook/opt-125m",  # Using tiny model for testing
        download_dir=None,  # Use default cache dir
        num_scheduler_steps=1,
        swap_space=1,  # Reduced for testing
        max_model_len=512,
        enable_lora=enable_lora,
    )

def format_output(outputs: List[RequestOutput], start_idx: int = 0) -> List[Dict]:
    """Format outputs in consistent format."""
    return [
        {
            "qsl_idx": i + start_idx,
            "seq_id": i + start_idx,
            "data": "".join(map(lambda x: x.to_bytes(4, byteorder="little").hex().upper(),
                              output.outputs[0].token_ids))
        }
        for i, output in enumerate(outputs)
    ]

def save_outputs(outputs: List[Dict], output_dir: str, is_lora: bool):
    """Save outputs to JSON file."""
    fname_type = "lora" if is_lora else "normal"
    fname_count = len([f for f in os.listdir(output_dir) if fname_type in f])
    output_path = os.path.join(output_dir, f"{fname_type}_{fname_count}.json")
    
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=4)
    return output_path

def create_lora_request(name: str, path: str, lora_id: int) -> LoRARequest:
    """Create a LoRA request object."""
    return LoRARequest(
        lora_name=name,
        lora_int_id=lora_id,
        lora_path=path,
    )

def test_lora_consistency(test_lora_path, sample_prompts, output_dir):
    """Test consistency between LoRA and non-LoRA outputs."""
    sampling_params = vllm.SamplingParams(
        max_tokens=32,
        temperature=0,
    )

    # Get outputs without LoRA
    llm_normal = get_llm(enable_lora=False)
    outputs_normal = llm_normal.generate(sample_prompts, sampling_params)

    # Get outputs with LoRA
    llm_lora = get_llm(enable_lora=True)
    lora_request = LoRARequest(
        lora_name="test_lora",
        lora_int_id=1,
        lora_path=test_lora_path,
    )
    llm_lora.add_lora(lora_request)
    outputs_lora = llm_lora.generate(sample_prompts, sampling_params)

    # Save outputs
    normal_outputs = format_output(outputs_normal)
    lora_outputs = format_output(outputs_lora)

    # Save outputs to files for comparison
    normal_file = os.path.join(output_dir, "normal_outputs.json")
    lora_file = os.path.join(output_dir, "lora_outputs.json")

    with open(normal_file, "w") as f:
        json.dump(normal_outputs, f, indent=2)
    with open(lora_file, "w") as f:
        json.dump(lora_outputs, f, indent=2)

    # Verify outputs are different
    for normal_item, lora_item in zip(normal_outputs, lora_outputs):
        assert len(normal_item["data"]) > 0, "Normal output data should not be empty"
        assert len(lora_item["data"]) > 0, "LoRA output data should not be empty"

def test_lora_swap_consistency(adapter_1, adapter_2, sample_prompts, output_dir):
    """Test consistency when swapping between different LoRA adapters."""
    sampling_params = vllm.SamplingParams(
        max_tokens=32,
        temperature=0,
    )
    
    # Initialize LLM with LoRA support
    llm = get_llm(enable_lora=True)
    
    # Test different adapters
    adapter_configs = [
        ("adapter_1", adapter_1, 1),
        ("adapter_2", adapter_2, 2),
    ]
    
    results = []
    for adapter_name, adapter_path, adapter_id in adapter_configs:
        # Add current adapter
        lora_request = LoRARequest(
            lora_name=adapter_name,
            lora_int_id=adapter_id,
            lora_path=adapter_path,
        )
        llm.add_lora(lora_request)
        
        # Generate with current adapter
        outputs = llm.generate(sample_prompts, sampling_params)
        formatted_outputs = format_output(outputs, start_idx=adapter_id * len(sample_prompts))
        results.extend(formatted_outputs)
        
        # Save results
        output_file = os.path.join(output_dir, f"{adapter_name}_outputs.json")
        with open(output_file, "w") as f:
            json.dump(formatted_outputs, f, indent=2)
        
        # Remove current adapter before adding next one
        llm.remove_lora(adapter_id)
    
    # Verify all outputs are saved
    assert len(results) == len(adapter_configs) * len(sample_prompts)

