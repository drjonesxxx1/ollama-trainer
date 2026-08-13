export type ModelArch = "llama3" | "qwen2.5" | "mistral" | "gemma2" | "deepseek" | "phi4" | "smollm" | "custom";

export type FineTuneMethod =
  | "qlora"
  | "lora_plus"
  | "dora"
  | "orpo"
  | "dpo"
  | "simpo"
  | "kto"
  | "galore"
  | "neftune"
  | "longlora";

export type PruningMethod =
  | "structured_layer"
  | "head_pruning"
  | "vocab_trim"
  | "laser_svd"
  | "wanda"
  | "magnitude_dropout";

export type MoEMethod =
  | "moefication"
  | "dare_ties"
  | "slerp"
  | "passthrough_franken"
  | "task_arithmetic"
  | "linear_average";

export type GGUFQuantType =
  | "Q4_K_M"
  | "Q4_K_S"
  | "Q5_K_M"
  | "Q5_K_S"
  | "Q8_0"
  | "IQ4_XS"
  | "IQ3_XXS"
  | "IQ2_XS"
  | "BF16"
  | "FP16";

export interface BaseModelInfo {
  id: string;
  name: string;
  huggingFaceId: string;
  ollamaName: string;
  parametersBillion: number;
  layers: number;
  hiddenDim: number;
  heads: number;
  kvHeads: number;
  vocabSize: number;
  defaultContext: number;
  architecture: ModelArch;
  baseSizeGb: number;
  q4SizeGb: number;
  description: string;
  recommendedFor4080Super: boolean;
}

export interface TrainingHyperparameters {
  // LoRA / PEFT
  lora_r: number;
  lora_alpha: number;
  lora_dropout?: number;
  target_modules?: string[];
  bias?: "none" | "all" | "lora_only";
  use_dora?: boolean;
  use_rslora?: boolean;
  
  // Optimizer & Scheduler
  batch_size: number;
  gradient_accumulation_steps: number;
  learning_rate: number;
  lr_scheduler?: "cosine" | "linear" | "constant" | "cosine_with_restarts";
  warmup_ratio?: number;
  warmup_steps?: number;
  weight_decay?: number;
  max_grad_norm?: number;
  optimizer?: "adamw_8bit" | "paged_adamw_8bit" | "adamw_torch" | "galore_adamw";
  
  // Training Duration & Precision
  epochs: number;
  max_steps?: number;
  max_seq_length: number;
  precision?: "bfloat16" | "float16";
  use_gradient_checkpointing?: boolean;
  use_unsloth_fast_backprop?: boolean;
  neftune_noise_alpha?: number;
  
  // Preference Alignment (for ORPO/DPO/SimPO)
  preference_beta?: number;
  simpo_gamma?: number;
}

export interface PruningConfig {
  enabled: boolean;
  pruneMethod?: PruningMethod;
  methods?: PruningMethod[];
  layerPruningRange: [number, number]; // e.g. prune layers 16 to 24
  targetLayersCount?: number;
  headsPrunePercentage?: number;
  headPruningRatio?: number; // 0.0 - 0.5
  vocabTrimTarget?: number;
  vocabTargetTokens?: number; // e.g. 32000 from 128000
  laserReductionRank?: number; // e.g. 32
  repairLoRASteps?: number;
  healingLoraSteps?: number;
}

export interface MoEConfig {
  enabled: boolean;
  method: MoEMethod;
  numExperts: number;
  topK: number;
  routerType: "softmax" | "sinkhorn" | "switch";
  expertSources: {
    name: string;
    modelId: string;
    weight: number;
    specialization: string;
  }[];
}

export interface MCPToolDeclaration {
  id: string;
  name: string;
  description: string;
  serverName: string;
  parametersSchema: {
    type: "object";
    properties: Record<string, { type: string; description: string; enum?: string[] }>;
    required?: string[];
  };
  sampleCallsCount?: number;
}

export interface TrainingDataSample {
  id: string;
  instruction: string;
  input?: string;
  output: string;
  system?: string;
  category?: string;
  difficulty?: string;
  toolCalls?: {
    name: string;
    arguments: Record<string, any>;
  }[];
  simulatedToolResult?: string;
  isMcpSample?: boolean;
}

export interface GGUFConfig {
  quantization: GGUFQuantType;
  contextLength: number;
  templateFormat?: "llama3" | "chatml" | "mistral" | "alpaca" | "deepseek" | "gemma";
  systemPrompt: string;
  stopTokens?: string[];
  temperature: number;
  top_p?: number;
  top_k?: number;
  repeat_penalty?: number;
  num_gpu_layers: number; // 999 for full 4080 Super offload
  threads?: number;
}

export interface HardwarePreset {
  name: string;
  vramGb: number;
  cudaCores: number;
  tensorCores: number;
  recommendedBatch: number;
  recommendedSeqLen: number;
  recommendedQuant: GGUFQuantType;
  notes: string;
}

export interface DistillationConfig {
  enabled: boolean;
  teacherModel: string;
  studentModel?: string;
  distillationType?: "response_generation" | "cot_reasoning" | "logit_kl" | "mcp_alignment";
  temperature: number;
  includeThoughtChain: boolean;
  distillDatasetSize?: number;
  samplesToGenerate?: number;
  distillationAlpha?: number;
}

export interface TrainingLogEntry {
  step: number;
  epoch: number;
  loss: number;
  evalLoss?: number;
  learningRate: number;
  gradNorm: number;
  vramUsedGb: number;
  tokensPerSec: number;
  sampleOutput?: string;
  timestamp: string;
}

export type ActiveTab =
  | "models"
  | "model"
  | "techniques"
  | "dataset"
  | "mcp"
  | "mcp_harness"
  | "distill"
  | "distillation"
  | "pruning"
  | "moe"
  | "moe_merge"
  | "gguf"
  | "train"
  | "training"
  | "deploy"
  | "ollama"
  | "arena";
