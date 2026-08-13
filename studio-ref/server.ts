import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

// Lazy initialize Gemini AI client
function getGeminiAI() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not configured. Please check your environment variables.");
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

// Health check endpoint
app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    hasApiKey: Boolean(process.env.GEMINI_API_KEY),
    timestamp: new Date().toISOString(),
  });
});

// 1. Generate Synthetic Training Data with Gemini
app.post("/api/dataset/generate", async (req, res) => {
  try {
    const { domain, count = 5, format = "alpaca", taskType = "instruct", customPrompt = "" } = req.body;
    const ai = getGeminiAI();

    const systemPrompt = `You are an elite LLM fine-tuning data engineer specializing in high-quality Unsloth and SOTA dataset synthesis.
Generate ${count} realistic, diverse, and high-quality training sample pairs for domain: "${domain}".
Task Type: "${taskType}".
Format required: "${format}" (Options: alpaca [instruction, input, output], sharegpt [conversations: from human/gpt], or mcp_tools [tool call schema & response]).
Ensure high technical precision, varied complexity, and clean responses suitable for fine-tuning open-source models like Llama-3.1, Qwen-2.5, and DeepSeek.
${customPrompt ? `Additional Instructions: ${customPrompt}` : ""}`;

    const response = await ai.models.generateContent({
      model: "gemini-3.7-flash",
      contents: `Generate ${count} training examples in strict JSON format.`,
      config: {
        systemInstruction: systemPrompt,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          description: "List of training data items",
          items: {
            type: Type.OBJECT,
            properties: {
              id: { type: Type.STRING },
              instruction: { type: Type.STRING },
              input: { type: Type.STRING },
              output: { type: Type.STRING },
              system: { type: Type.STRING },
              category: { type: Type.STRING },
              difficulty: { type: Type.STRING },
            },
            required: ["instruction", "output"],
          },
        },
      },
    });

    const items = JSON.parse(response.text || "[]");
    res.json({ success: true, count: items.length, data: items });
  } catch (error: any) {
    console.error("Error generating dataset:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to generate dataset" });
  }
});

// 2. Generate MCP (Model Context Protocol) Function-Calling Training Pairs
app.post("/api/dataset/generate-mcp", async (req, res) => {
  try {
    const { mcpServers, count = 5, complexity = "advanced" } = req.body;
    const ai = getGeminiAI();

    const systemInstruction = `You are a specialist in MCP (Model Context Protocol) and Function-Calling fine-tuning for local models running on Ollama/vLLM.
You are given the following MCP tool declarations:
${JSON.stringify(mcpServers, null, 2)}

Generate ${count} realistic user queries that require one or multiple tool invocations using the declared tools, followed by synthetic tool execution results and final assistant reasoning.
Complexity level: ${complexity}.
Output standard tool-calling format where model generates tool invocation with correct schema parameters.`;

    const response = await ai.models.generateContent({
      model: "gemini-3.7-flash",
      contents: "Generate MCP tool calling training examples.",
      config: {
        systemInstruction,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              id: { type: Type.STRING },
              userQuery: { type: Type.STRING },
              toolCalls: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    name: { type: Type.STRING },
                    arguments: { type: Type.OBJECT },
                  },
                  required: ["name"],
                },
              },
              simulatedToolResult: { type: Type.STRING },
              assistantResponse: { type: Type.STRING },
              thoughtChain: { type: Type.STRING },
            },
            required: ["userQuery", "toolCalls", "assistantResponse"],
          },
        },
      },
    });

    const data = JSON.parse(response.text || "[]");
    res.json({ success: true, count: data.length, data });
  } catch (error: any) {
    console.error("Error generating MCP dataset:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to generate MCP dataset" });
  }
});

// 3. Teacher-to-Student Model Distillation Pair Generator
app.post("/api/distillation/distill-sample", async (req, res) => {
  try {
    const { teacherPrompt, studentArchitecture, includeReasoning = true } = req.body;
    const ai = getGeminiAI();

    const systemInstruction = `You are acting as an elite Teacher model distilling reasoning, knowledge, and structured responses to a compact student model (${studentArchitecture || "8B Student"}).
Provide a comprehensive, high-quality reference response. ${includeReasoning ? "Include a <think>...</think> chain-of-thought section before the final answer." : ""}`;

    const response = await ai.models.generateContent({
      model: "gemini-3.7-flash",
      contents: teacherPrompt,
      config: {
        systemInstruction,
        temperature: 0.7,
      },
    });

    res.json({
      success: true,
      teacherResponse: response.text || "",
      model: "gemini-3.7-flash (Teacher Mode)",
    });
  } catch (error: any) {
    console.error("Error in distillation:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to generate teacher response" });
  }
});

// 4. AI Training Optimization Advisor
app.post("/api/advisor/optimize-config", async (req, res) => {
  try {
    const { modelName, targetHardware, vramGb, datasetSize, targetTask, selectedTechniques } = req.body;
    const ai = getGeminiAI();

    const prompt = `Analyze this LLM fine-tuning setup and recommend the optimal hyperparameters and Unsloth/GGUF quantization strategy:
- Base Model: ${modelName}
- Target GPU: ${targetHardware} (${vramGb} GB VRAM)
- Dataset Size: ${datasetSize} samples
- Goal/Task: ${targetTask}
- Selected Techniques: ${JSON.stringify(selectedTechniques)}

Provide:
1. Exact LoRA rank (r), alpha, target_modules for Unsloth
2. Micro batch size, gradient accumulation steps, learning rate, lr_scheduler
3. Exact VRAM estimation during training & inference
4. GGUF Quantization recommendation (e.g. Q4_K_M, IQ4_XS) with explanation
5. Key warnings or recommendations for Windows RTX 4080 Super`;

    const response = await ai.models.generateContent({
      model: "gemini-3.7-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            recommendedLoRA_r: { type: Type.INTEGER },
            recommendedLoRA_alpha: { type: Type.INTEGER },
            batchSize: { type: Type.INTEGER },
            gradAccumSteps: { type: Type.INTEGER },
            learningRate: { type: Type.STRING },
            trainingVramEstimateGb: { type: Type.NUMBER },
            inferenceVramEstimateGb: { type: Type.NUMBER },
            recommendedQuantization: { type: Type.STRING },
            fitProbabilityPercent: { type: Type.INTEGER },
            tips: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            modelfileParameters: {
              type: Type.OBJECT,
              properties: {
                temperature: { type: Type.NUMBER },
                top_p: { type: Type.NUMBER },
                num_ctx: { type: Type.INTEGER },
                num_gpu: { type: Type.INTEGER },
              },
            },
          },
          required: [
            "recommendedLoRA_r",
            "batchSize",
            "gradAccumSteps",
            "learningRate",
            "trainingVramEstimateGb",
            "recommendedQuantization",
            "fitProbabilityPercent",
            "tips",
          ],
        },
      },
    });

    const parsed = JSON.parse(response.text || "{}");
    res.json({ success: true, config: parsed });
  } catch (error: any) {
    console.error("Error optimizing config:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to generate optimization config" });
  }
});

// 5. Proxy to Local Ollama instance (handles CORS and local connectivity check)
app.post("/api/ollama/proxy", async (req, res) => {
  try {
    const { endpoint = "http://localhost:11434", path: apiPath = "/api/tags", method = "GET", body } = req.body;
    const url = `${endpoint.replace(/\/$/, "")}${apiPath.startsWith("/") ? "" : "/"}${apiPath}`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);

    const fetchOptions: RequestInit = {
      method,
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
    };
    if (body && (method === "POST" || method === "PUT")) {
      fetchOptions.body = JSON.stringify(body);
    }

    const response = await fetch(url, fetchOptions);
    clearTimeout(timeoutId);

    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      const data = await response.json();
      res.json({ success: true, status: response.status, data });
    } else {
      const text = await response.text();
      res.json({ success: true, status: response.status, text });
    }
  } catch (error: any) {
    res.status(502).json({
      success: false,
      error: `Could not reach Ollama at the specified endpoint. Ensure Ollama is running (e.g. 'ollama serve' or default service).`,
      details: error.message,
    });
  }
});

// Setup Vite middleware for full-stack SPA development
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Ollama Unsloth Studio server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
