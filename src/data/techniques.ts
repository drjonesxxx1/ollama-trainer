export interface TechniqueDetail {
  id: string;
  name: string;
  category: "finetune" | "prune" | "moe" | "quant" | "alignment";
  tagline: string;
  description: string;
  paperReference: string;
  memorySavings: string;
  speedMultiplier: string;
  recommendedFor4080: boolean;
  unslothSupported: boolean;
  codeSnippet: string;
  parameters: {
    name: string;
    label: string;
    type: "number" | "select" | "boolean" | "text";
    default: any;
    options?: string[];
    description: string;
  }[];
}

export const SOTA_TECHNIQUES: TechniqueDetail[] = [
  {
    id: "qlora",
    name: "Unsloth Fast QLoRA (4-bit NF4)",
    category: "finetune",
    tagline: "Ultra-fast parameter efficient fine-tuning with 4-bit NormalFloat quantization",
    description: "Quantizes base weights to 4-bit NormalFloat (NF4) with double quantization and trains 16-bit LoRA adapter matrices via custom Triton kernels. Reduces VRAM by up to 80% while retaining full 16-bit accuracy.",
    paperReference: "Dettmers et al., 2023 (QLoRA) & Unsloth AI",
    memorySavings: "80% VRAM reduction",
    speedMultiplier: "2.2x - 5.0x faster",
    recommendedFor4080: true,
    unslothSupported: true,
    codeSnippet: `model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)`,
    parameters: [
      { name: "lora_r", label: "LoRA Rank (r)", type: "number", default: 16, description: "Dimension of low-rank update matrices (8, 16, 32, 64)" },
      { name: "lora_alpha", label: "LoRA Alpha", type: "number", default: 16, description: "Scaling factor (commonly 1x or 2x of rank r)" },
      { name: "use_gradient_checkpointing", label: "Unsloth Fast Gradient Checkpointing", type: "boolean", default: true, description: "Offloads activations to save 40% memory with zero speed penalty" },
    ],
  },
  {
    id: "dora",
    name: "DoRA (Weight-Decomposed Low-Rank Adaptation)",
    category: "finetune",
    tagline: "Decomposes weights into magnitude and direction for full-fine-tuning parity",
    description: "Decomposes pre-trained weights into magnitude vectors and directional matrices. LoRA is applied exclusively to the directional component, matching or exceeding full fine-tuning performance without extra inference cost.",
    paperReference: "Liu et al., 2024 (DoRA: Weight-Decomposed Low-Rank Adaptation)",
    memorySavings: "75% VRAM reduction",
    speedMultiplier: "1.8x faster",
    recommendedFor4080: true,
    unslothSupported: true,
    codeSnippet: `model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    use_dora=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)`,
    parameters: [
      { name: "use_dora", label: "Enable DoRA Decomposition", type: "boolean", default: true, description: "Enable magnitude/directional weight split" },
      { name: "lora_r", label: "Directional Rank", type: "number", default: 16, description: "Rank for the directional matrix" },
    ],
  },
  {
    id: "orpo",
    name: "ORPO (Odds Ratio Preference Optimization)",
    category: "alignment",
    tagline: "Single-step preference alignment & SFT without a reference model",
    description: "Integrates odds-ratio penalty directly into the supervised cross-entropy loss function. Eliminates the need for a separate reference model or two-stage SFT+DPO pipeline, halving VRAM requirements.",
    paperReference: "Hong et al., 2024 (ORPO: Monolithic Preference Optimization)",
    memorySavings: "50% VRAM saving vs DPO",
    speedMultiplier: "2.0x faster than SFT+DPO",
    recommendedFor4080: true,
    unslothSupported: true,
    codeSnippet: `from trl import ORPOTrainer, ORPOConfig

orpo_trainer = ORPOTrainer(
    model=model,
    args=ORPOConfig(
        beta=0.1,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        max_prompt_length=1024,
        max_length=2048,
    ),
    train_dataset=dataset,
)`,
    parameters: [
      { name: "preference_beta", label: "Odds Ratio Beta (β)", type: "number", default: 0.1, description: "Weight of the preference penalty in ORPO loss (0.05 - 0.2)" },
    ],
  },
  {
    id: "galore",
    name: "GaLore (Gradient Low-Rank Projection)",
    category: "finetune",
    tagline: "Memory-efficient full-parameter training via gradient subspace projection",
    description: "Allows full parameter training of 7B-14B models on 16GB VRAM by projecting optimizer state gradients into low-rank subspaces, slashing optimizer memory by up to 65.5%.",
    paperReference: "Zhao et al., 2024 (GaLore: Gradient Low-Rank Projection)",
    memorySavings: "65% optimizer memory reduction",
    speedMultiplier: "1.2x",
    recommendedFor4080: true,
    unslothSupported: false,
    codeSnippet: `from galore_torch import GaLoreAdamW8bit

optimizer = GaLoreAdamW8bit(
    model.parameters(),
    lr=1e-5,
    rank=128,
    update_proj_gap=200,
    scale=0.25,
)`,
    parameters: [
      { name: "galore_rank", label: "Gradient Subspace Rank", type: "number", default: 128, description: "Projection rank for gradients" },
      { name: "update_proj_gap", label: "Projection Update Frequency", type: "number", default: 200, description: "Steps between SVD subspace updates" },
    ],
  },
  {
    id: "neftune",
    name: "NEFTune (Noisy Embedding Fine-Tuning)",
    category: "finetune",
    tagline: "Injects uniform noise into embeddings to boost generalizability and prevent overfitting",
    description: "Adds scaled uniform random noise to input token embeddings during training. Proven to boost AlpacaEval and conversational benchmark scores by 5-15% with zero extra VRAM.",
    paperReference: "Jain et al., 2023 (NEFTune: Noisy Embeddings Improve Instruction Finetuning)",
    memorySavings: "0% (Zero overhead)",
    speedMultiplier: "1.0x",
    recommendedFor4080: true,
    unslothSupported: true,
    codeSnippet: `trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    neftune_noise_alpha=5,
)`,
    parameters: [
      { name: "neftune_noise_alpha", label: "Noise Alpha Scale", type: "number", default: 5, description: "Magnitude of uniform noise added to embeddings (typically 5 to 15)" },
    ],
  },
  {
    id: "structured_layer",
    name: "ShortGPT Structured Layer Pruning (Fat Shaving)",
    category: "prune",
    tagline: "Removes redundant hidden layers based on angular similarity metric",
    description: "Calculates the cosine angular distance of representations between consecutive transformer layers. Redundant layers (often layers 14-22 in a 32-layer model) are trimmed, reducing model parameters by 25-35% with minimal accuracy loss.",
    paperReference: "Men et al., 2024 (ShortGPT: Layers in Large Language Models are More Redundant Than You Think)",
    memorySavings: "25-35% permanent size reduction",
    speedMultiplier: "1.35x faster inference",
    recommendedFor4080: true,
    unslothSupported: true,
    codeSnippet: `# Pruning redundant middle layers from 32 layers down to 24 layers
pruned_layers = [i for i in range(32) if i not in range(16, 24)]
model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in pruned_layers])
model.config.num_hidden_layers = len(pruned_layers)
# Followed by 100-step healing LoRA adapter`,
    parameters: [
      { name: "layers_to_prune", label: "Pruning Range (Start - End)", type: "text", default: "16-23", description: "Indices of transformer layers to excise" },
      { name: "repair_steps", label: "Healing LoRA Steps", type: "number", default: 100, description: "Short LoRA fine-tune steps to restore perplexity" },
    ],
  },
  {
    id: "vocab_trim",
    name: "Vocabulary Trimmer (Shave 1GB of Embedding Fat)",
    category: "prune",
    tagline: "Trims unused multilingual and rare tokens from 128k tokenizer down to 32k",
    description: "Modern tokenizers (Llama 3.1 & Qwen 2.5) allocate 128k-152k tokens, consuming over 1.2 GB VRAM in the embedding table alone. Trimming down to target domain tokens shrinks the GGUF file substantially.",
    paperReference: "TokenCraft / CompactLLM 2024",
    memorySavings: "800MB - 1.4GB disk & VRAM savings",
    speedMultiplier: "1.15x faster generation",
    recommendedFor4080: true,
    unslothSupported: true,
    codeSnippet: `# Shrink embedding matrix and lm_head
kept_token_ids = get_frequent_tokens(dataset, target_size=32000)
model.resize_token_embeddings(len(kept_token_ids))`,
    parameters: [
      { name: "target_vocab", label: "Target Vocabulary Size", type: "number", default: 32000, description: "Size to condense the 128k/152k vocabulary to" },
    ],
  },
  {
    id: "moefication",
    name: "MoEfication & FFN Clustering (Adding Experts)",
    category: "moe",
    tagline: "Converts a dense 8B model into an 8x8B Mixture of Experts with router",
    description: "Splits the dense MLP/feed-forward layers into specialized expert clusters via k-means weight clustering, training a top-2 gating router. Delivers higher representational capacity while keeping active inference compute fixed.",
    paperReference: "Zhang et al., 2022 (MoEfication: Transformer Feed-forward Layers are Sparse Experts)",
    memorySavings: "Inference compute equals 1 expert",
    speedMultiplier: "MoE Sparse routing",
    recommendedFor4080: true,
    unslothSupported: true,
    codeSnippet: `# Upcycle dense model to Mixture of Experts
from mergekit.moe import MoEBuilder

builder = MoEBuilder(
    base_model="unsloth/Meta-Llama-3.1-8B-Instruct",
    num_experts=4,
    top_k=2,
    router_type="softmax",
)
builder.build_moe_architecture()`,
    parameters: [
      { name: "num_experts", label: "Number of Experts", type: "number", default: 4, description: "Total expert blocks (e.g. 4 or 8)" },
      { name: "top_k", label: "Top-K Active Experts", type: "number", default: 2, description: "Number of experts activated per token (usually 1 or 2)" },
    ],
  },
  {
    id: "dare_ties",
    name: "DARE-TIES Model Merging (MergeKit)",
    category: "moe",
    tagline: "Drops redundant delta parameters and resolves sign conflicts across fine-tunes",
    description: "Merges multiple specialized models (e.g. your Coding fine-tune + your MCP Tool fine-tune) by dropping 90% of insignificant weight deltas and rescaling the rest with Task-Informed Energy Sign resolution.",
    paperReference: "Yu et al., 2024 (Language Models are Super Mario: DARE)",
    memorySavings: "Combines models with zero training cost",
    speedMultiplier: "Instant merge",
    recommendedFor4080: true,
    unslothSupported: true,
    codeSnippet: `merge_method: dare_ties
base_model: unsloth/Meta-Llama-3.1-8B-Instruct
models:
  - model: ./my-coding-adapter-merged
    parameters:
      weight: 0.6
      density: 0.2
  - model: ./my-mcp-tool-adapter-merged
    parameters:
      weight: 0.4
      density: 0.2
dtype: bfloat16`,
    parameters: [
      { name: "density", label: "Weight Delta Density", type: "number", default: 0.2, description: "Fraction of extreme weights to retain (0.1 to 0.4)" },
    ],
  },
];
