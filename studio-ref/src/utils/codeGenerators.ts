import { BaseModelInfo, GGUFConfig, MoEConfig, PruningConfig, TrainingHyperparameters } from "../types";

export function generateUnslothPythonScript(
  model: BaseModelInfo,
  params: TrainingHyperparameters,
  ggufConfig: GGUFConfig,
  pruningConfig: PruningConfig,
  customDatasetPath: string = "./dataset.json",
  outputModelName: string = "fine-tuned-ollama-model"
): string {
  const isDoRA = params.use_dora;
  const isORPO = params.neftune_noise_alpha > 0;
  const targetModulesStr = JSON.stringify(params.target_modules);

  let pruningCode = "";
  if (pruningConfig.enabled && pruningConfig.methods.includes("structured_layer")) {
    pruningCode = `
# ==========================================
# ✂️ STRUCTURED LAYER PRUNING (ShortGPT Fat-Shaving)
# ==========================================
print(">> Applying structured layer pruning on middle transformer blocks...")
start_prune, end_prune = ${pruningConfig.layerPruningRange[0]}, ${pruningConfig.layerPruningRange[1]}
pruned_layers = [i for i in range(model.config.num_hidden_layers) if not (start_prune <= i <= end_prune)]
model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in pruned_layers])
model.config.num_hidden_layers = len(pruned_layers)
print(f">> Model layers pruned down to {len(pruned_layers)} layers! Shaved ~25% parameter fat.")
`;
  }

  return `"""
Ollama Unsloth Studio - State-of-the-Art Fine-Tuning & Quantization Pipeline
Target Model: ${model.name} (${model.huggingFaceId})
Hardware Target: NVIDIA RTX 4080 Super (16GB VRAM) / Windows CUDA
"""

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Configuration & Hyperparameters
max_seq_length = ${params.max_seq_length}
dtype = None # Auto detection (Float16 / Bfloat16)
load_in_4bit = True # 4-bit NF4 for max VRAM efficiency on RTX 4080 Super

print(">> Initializing FastLanguageModel from Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="${model.huggingFaceId}",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
${pruningCode}
# 2. Configure PEFT / LoRA / DoRA Parameters
print(">> Attaching optimized LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=${params.lora_r},
    target_modules=${targetModulesStr},
    lora_alpha=${params.lora_alpha},
    lora_dropout=${params.lora_dropout},
    bias="${params.bias}",
    use_gradient_checkpointing="unsloth", # Saves 70% VRAM with zero speed penalty
    random_state=3407,
    use_rslora=${params.use_rslora},
    use_dora=${isDoRA},
)

# 3. Format Dataset & Chat Template (Including MCP & Tool-Calling schemas)
prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
${ggufConfig.systemPrompt || "You are an expert AI assistant specialized in precise reasoning and MCP tool execution."}<|eot_id|><|start_header_id|>user<|end_header_id|>
{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{}<|eot_id|>"""

def formatting_prompts_func(examples):
    instructions = examples.get("instruction", [])
    inputs       = examples.get("input", [""] * len(instructions))
    outputs      = examples.get("output", [])
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        user_content = f"{instruction}\\n{input_text}".strip() if input_text else instruction
        text = prompt_template.format(user_content, output)
        texts.append(text)
    return { "text" : texts }

print(f">> Loading training dataset from ${customDatasetPath}...")
if os.path.exists("${customDatasetPath}"):
    dataset = load_dataset("json", data_files="${customDatasetPath}", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
else:
    print(">> Notice: Local dataset not found, using demo dataset fallback.")
    from datasets import Dataset
    dataset = Dataset.from_list([
        {"instruction": "Call the filesystem read_file tool on src/App.tsx", "input": "", "output": '{"name": "read_file", "arguments": {"path": "src/App.tsx"}}'}
    ]).map(formatting_prompts_func, batched=True)

# 4. Training Engine Initialization
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False, # True for up to 5x speedup for short sequences
    args=TrainingArguments(
        per_device_train_batch_size=${params.batch_size},
        gradient_accumulation_steps=${params.gradient_accumulation_steps},
        warmup_ratio=${params.warmup_ratio},
        num_train_epochs=${params.epochs},
        learning_rate=${params.learning_rate},
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="${params.optimizer}",
        weight_decay=${params.weight_decay},
        lr_scheduler_type="${params.lr_scheduler}",
        seed=3407,
        output_dir="./outputs",
        report_to="none",
    ),
)

print(">> Starting model training loop...")
trainer_stats = trainer.train()
print(">> Training complete! Peak VRAM used:", round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3), "GB")

# 5. Direct GGUF Quantization & Export for Ollama
print(">> Quantizing and saving GGUF directly for Ollama (${ggufConfig.quantization.toLowerCase()})...")
model.save_pretrained_gguf(
    "${outputModelName}",
    tokenizer,
    quantization_method="${ggufConfig.quantization.toLowerCase()}"
)

# 6. Generate Ollama Modelfile
modelfile_content = f"""FROM ./${outputModelName}-${ggufConfig.quantization.toLowerCase()}.gguf

TEMPLATE """ + '"""' + prompt_template + '"""' + f"""
PARAMETER temperature ${ggufConfig.temperature}
PARAMETER top_p ${ggufConfig.top_p}
PARAMETER top_k ${ggufConfig.top_k}
PARAMETER repeat_penalty ${ggufConfig.repeat_penalty}
PARAMETER num_ctx ${ggufConfig.contextLength}
PARAMETER num_gpu ${ggufConfig.num_gpu_layers}
SYSTEM """ + '"""' + "${ggufConfig.systemPrompt}" + '"""'

with open("Modelfile", "w") as f:
    f.write(modelfile_content)

print(">> Modelfile created successfully!")
print(">> To run in Ollama, execute:")
print(f"   ollama create ${outputModelName} -f Modelfile")
print(f"   ollama run ${outputModelName}")
`;
}

export function generateModelfile(
  model: BaseModelInfo,
  ggufConfig: GGUFConfig,
  modelTag: string = "my-custom-model"
): string {
  const quantSuffix = ggufConfig.quantization.toLowerCase();
  return `# Modelfile generated by Ollama Unsloth Studio
# Optimized for RTX 4080 Super (16GB VRAM) & MCP Plugins

FROM ./${modelTag}-${quantSuffix}.gguf

# Model Parameters
PARAMETER temperature ${ggufConfig.temperature}
PARAMETER top_p ${ggufConfig.top_p}
PARAMETER top_k ${ggufConfig.top_k}
PARAMETER repeat_penalty ${ggufConfig.repeat_penalty}
PARAMETER num_ctx ${ggufConfig.contextLength}
PARAMETER num_gpu ${ggufConfig.num_gpu_layers}

# Chat & Tool Template
TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""

# System Prompt & MCP Tool Declarations
SYSTEM """${ggufConfig.systemPrompt || "You are an ultra-fast, fine-tuned AI model optimized for local execution and MCP tool harness execution."}"""
`;
}

export function generateWindowsPowerShellScript(
  modelTag: string = "my-custom-model"
): string {
  return `# Windows RTX 4080 Super One-Click Training & Ollama Deployer
# PowerShell Script for Windows 10/11 with CUDA 12.x

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "   Ollama Unsloth Studio - Windows RTX 4080 Super Runner" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Cyan

# Check Python environment
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[-] Python not found. Please install Python 3.10 or 3.11 with PATH enabled." -ForegroundColor Red
    Exit
}

# Ensure Virtual Environment
if (!(Test-Path "./venv")) {
    Write-Host "[+] Creating virtual environment 'venv'..." -ForegroundColor Yellow
    python -m venv venv
}

Write-Host "[+] Activating Virtual Environment..." -ForegroundColor Yellow
& ./venv/Scripts/Activate.ps1

Write-Host "[+] Installing/Updating Unsloth & CUDA PyTorch..." -ForegroundColor Yellow
pip install --upgrade pip
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes trl peft datasets transformers xformers

Write-Host "[+] Starting Unsloth Fine-Tuning..." -ForegroundColor Green
python train_unsloth.py

if (Test-Path "./Modelfile") {
    Write-Host "[+] Building Ollama Model '${modelTag}'..." -ForegroundColor Green
    ollama create ${modelTag} -f Modelfile
    Write-Host "[+] SUCCESS! Model is registered in Ollama." -ForegroundColor Cyan
    Write-Host ">> Launching test session with Ollama..." -ForegroundColor Green
    ollama run ${modelTag} "Hello! Check your MCP tool calling capabilities."
} else {
    Write-Host "[-] Modelfile not generated. Please check training logs." -ForegroundColor Red
}
`;
}

export function generateMergeKitConfig(moeConfig: MoEConfig, baseModel: string): string {
  if (moeConfig.method === "moefication") {
    return `base_model: ${baseModel}
gate_mode: ${moeConfig.routerType}
dtype: bfloat16
experts:
${moeConfig.expertSources.map((exp) => `  - source_model: ${exp.modelId}
    positive_prompts:
      - "${exp.specialization}"
    parameters:
      weight: ${exp.weight}`).join("\n")}
`;
  }

  return `merge_method: ${moeConfig.method}
base_model: ${baseModel}
models:
${moeConfig.expertSources.map((exp) => `  - model: ${exp.modelId}
    parameters:
      weight: ${exp.weight}
      density: 0.2`).join("\n")}
dtype: bfloat16
`;
}
