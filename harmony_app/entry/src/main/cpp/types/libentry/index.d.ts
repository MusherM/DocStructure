import { resourceManager } from '@kit.LocalizationKit';

export interface InferenceResult {
  ok: boolean;
  text: string;
  gear: string;
  errorCode: string;
  message: string;
}

export const runInference: (
  resourceManager: resourceManager.ResourceManager,
  bgraPixels: Uint8Array,
  scaledWidth: number,
  scaledHeight: number,
  gear: string,
  maxTokens: number
) => Promise<InferenceResult>;

export const unload: () => void;

