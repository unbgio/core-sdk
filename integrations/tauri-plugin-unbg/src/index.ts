export type ModelAlias = "auto" | "fast" | "quality" | "rmbg-1.4" | "rmbg-2.0";
export type OnnxVariant = "auto" | "fp16" | "fp32" | "quantized";
export type ExecutionProvider = "auto" | "gpu" | "cpu";
export type GpuBackend = "auto" | "directml" | "cuda" | "coreml" | "metal";

export interface RemoveBackgroundRequest {
  imageBytes: number[];
  width: number;
  height: number;
  model?: ModelAlias;
  maxInferencePixels?: number;
  executionProvider?: ExecutionProvider;
  gpuBackend?: GpuBackend;
  benchmarkProvider?: boolean;
  onnxVariant?: OnnxVariant;
  modelDir?: string;
}

export interface RemoveBackgroundResponse {
  modelUsed: string;
  width: number;
  height: number;
  maskPng: number[];
  providerSelected: string;
  backendSelected?: string | null;
  fallbackUsed: boolean;
}

export type InvokeLike = <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;

export const TAURI_UNBG_COMMANDS_V1 = {
  removeBackground: "plugin:unbg|tauri_remove_background_command"
} as const;

export type RemoveBackgroundRequestV1 = RemoveBackgroundRequest;
export type RemoveBackgroundResponseV1 = RemoveBackgroundResponse;

/**
 * Typed wrapper around the Rust-side Tauri command.
 */
export async function removeBackground(
  invoke: InvokeLike,
  request: RemoveBackgroundRequest
): Promise<RemoveBackgroundResponse> {
  return invoke<RemoveBackgroundResponse>(TAURI_UNBG_COMMANDS_V1.removeBackground, {
    request: {
      model: "auto",
      benchmarkProvider: true,
      ...request
    }
  });
}

export const MODEL_ALIASES: readonly ModelAlias[] = [
  "auto",
  "fast",
  "quality",
  "rmbg-1.4",
  "rmbg-2.0"
];
