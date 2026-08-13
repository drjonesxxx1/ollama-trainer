import { BaseModelInfo, GGUFConfig, GGUFQuantType, MoEConfig, PruningConfig, TrainingHyperparameters } from "../types";

export interface VRAMCalculationResult {
  baseModelVramGb: number;
  loraOverheadGb: number;
  activationsGb: number;
  optimizerStateGb: number;
  kvCacheGb: number;
  totalTrainingVramGb: number;
  fitsIn4080Super: boolean;
  utilizationPercent: number;
  recommendedBatchSize: number;
  recommendedGradAccum: number;
  maxRecommendedContext: number;
  warnings: string[];
}

export function calculateVRAMFootprint(
  model: BaseModelInfo,
  params: TrainingHyperparameters,
  targetVramGb: number = 16.0
): VRAMCalculationResult {
  const warnings: string[] = [];

  // Base model in 4-bit NF4 (bitsandbytes / Unsloth)
  // 4-bit is ~0.55 bytes per parameter including quantization scale factors
  const baseModelVramGb = (model.parametersBillion * 1e9 * 0.55) / (1024 * 1024 * 1024);

  // LoRA rank overhead
  const numAdaptedModules = (params.target_modules && params.target_modules.length) || 7;
  const numLayers = model.layers;
  const hiddenDim = model.hiddenDim;
  const loraParams = 2 * numAdaptedModules * numLayers * hiddenDim * params.lora_r;
  // LoRA weights + gradients in fp32/bf16
  const loraOverheadGb = (loraParams * 4 * 2) / (1024 * 1024 * 1024);

  // Optimizer state: Paged AdamW 8-bit uses 2 bytes per trainable parameter
  const optimizerStateGb = (loraParams * 2) / (1024 * 1024 * 1024);

  // KV cache + activations: with Unsloth fast backprop + gradient checkpointing
  const seqLength = params.max_seq_length || 4096;
  const batchSize = params.batch_size || 1;
  
  let activationFactor = 0.00000000035;
  if (params.use_unsloth_fast_backprop) {
    activationFactor *= 0.35; // Unsloth cuts activation memory drastically
  }
  if (params.use_gradient_checkpointing) {
    activationFactor *= 0.45;
  }

  const activationsGb = (batchSize * seqLength * numLayers * hiddenDim * activationFactor);

  // KV cache overhead for eval/inference
  const kvCacheGb = (2 * numLayers * (model.kvHeads || 8) * (hiddenDim / (model.heads || 32)) * seqLength * 2) / (1024 * 1024 * 1024);

  const totalTrainingVramGb = Number(
    (baseModelVramGb + loraOverheadGb + optimizerStateGb + activationsGb + 0.8).toFixed(2) // 0.8GB CUDA baseline runtime
  );

  const fitsIn4080Super = totalTrainingVramGb <= targetVramGb;
  const utilizationPercent = Math.min(100, Math.round((totalTrainingVramGb / targetVramGb) * 100));

  let recommendedBatchSize = 2;
  let recommendedGradAccum = 4;
  let maxRecommendedContext = 32768;

  if (model.parametersBillion > 13) {
    recommendedBatchSize = 1;
    recommendedGradAccum = 8;
    maxRecommendedContext = 8192;
    if (totalTrainingVramGb > 15.5) {
      warnings.push("14B models on 16GB VRAM require micro-batch size 1 and gradient accumulation.");
    }
  } else if (model.parametersBillion > 30) {
    warnings.push("30B+ models exceed single 16GB VRAM for training. Use Student-Teacher Distillation or Multi-GPU.");
  }

  if (seqLength > 16384 && !params.use_gradient_checkpointing) {
    warnings.push("High context length (>16k) requires Gradient Checkpointing to avoid Out-Of-Memory (OOM).");
  }

  return {
    baseModelVramGb: Number(baseModelVramGb.toFixed(2)),
    loraOverheadGb: Number(loraOverheadGb.toFixed(2)),
    activationsGb: Number(activationsGb.toFixed(2)),
    optimizerStateGb: Number(optimizerStateGb.toFixed(2)),
    kvCacheGb: Number(kvCacheGb.toFixed(2)),
    totalTrainingVramGb,
    fitsIn4080Super,
    utilizationPercent,
    recommendedBatchSize,
    recommendedGradAccum,
    maxRecommendedContext,
    warnings,
  };
}

export function calculateHardwareCompatibility(
  model: BaseModelInfo,
  params: TrainingHyperparameters,
  ggufConfig?: GGUFConfig,
  pruningConfig?: PruningConfig,
  moeConfig?: MoEConfig
): VRAMCalculationResult {
  const result = calculateVRAMFootprint(model, params, 16.0);

  // Apply pruning reductions
  if (pruningConfig && pruningConfig.enabled) {
    const prunedLayers = Math.max(0, pruningConfig.layerPruningRange[1] - pruningConfig.layerPruningRange[0] + 1);
    const reductionRatio = (model.layers - prunedLayers) / model.layers;
    result.baseModelVramGb = Number((result.baseModelVramGb * reductionRatio).toFixed(2));
    result.totalTrainingVramGb = Number((result.totalTrainingVramGb - 1.2).toFixed(2));
    result.utilizationPercent = Math.min(100, Math.round((result.totalTrainingVramGb / 16.0) * 100));
  }

  return result;
}

export function getGGUFSizeEstimate(paramsBillion: number, quant: GGUFQuantType): { sizeGb: number; bitsPerWeight: number; ramRequiredGb: number } {
  const quantMap: Record<GGUFQuantType, { bpw: number; name: string }> = {
    Q4_K_M: { bpw: 4.5, name: "Q4_K_M (Optimal balance)" },
    Q4_K_S: { bpw: 4.1, name: "Q4_K_S (Compact 4-bit)" },
    Q5_K_M: { bpw: 5.5, name: "Q5_K_M (High precision)" },
    Q5_K_S: { bpw: 5.1, name: "Q5_K_S" },
    Q8_0: { bpw: 8.5, name: "Q8_0 (Near lossless)" },
    IQ4_XS: { bpw: 4.25, name: "IQ4_XS (Importance Matrix 4-bit)" },
    IQ3_XXS: { bpw: 3.06, name: "IQ3_XXS (Ultra slim 3-bit)" },
    IQ2_XS: { bpw: 2.31, name: "IQ2_XS (Extreme 2-bit)" },
    BF16: { bpw: 16.0, name: "BF16 (Unquantized)" },
    FP16: { bpw: 16.0, name: "FP16 (Unquantized)" },
  };

  const info = quantMap[quant] || { bpw: 4.5, name: "Q4_K_M" };
  const sizeGb = Number(((paramsBillion * 1e9 * (info.bpw / 8)) / (1024 * 1024 * 1024)).toFixed(2));
  const ramRequiredGb = Number((sizeGb + 1.2).toFixed(2)); // +1.2GB for context & KV cache in Ollama

  return {
    sizeGb,
    bitsPerWeight: info.bpw,
    ramRequiredGb,
  };
}
