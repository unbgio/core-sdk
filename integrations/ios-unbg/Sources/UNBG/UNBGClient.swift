import Foundation

public struct UNBGRemoveBackgroundRequest {
    public let imageBytes: Data
    public let width: UInt32
    public let height: UInt32
    public let model: String
    public let onnxVariant: String?
    public let executionProvider: String?
    public let gpuBackend: String?
    public let benchmarkProvider: Bool?
    public let modelDir: String?
    public let maxInferencePixels: UInt32?

    public init(
        imageBytes: Data,
        width: UInt32,
        height: UInt32,
        model: String = "auto",
        onnxVariant: String? = "fp16",
        executionProvider: String? = "auto",
        gpuBackend: String? = "auto",
        benchmarkProvider: Bool? = true,
        modelDir: String? = nil,
        maxInferencePixels: UInt32? = 1_500_000
    ) {
        self.imageBytes = imageBytes
        self.width = width
        self.height = height
        self.model = model
        self.onnxVariant = onnxVariant
        self.executionProvider = executionProvider
        self.gpuBackend = gpuBackend
        self.benchmarkProvider = benchmarkProvider
        self.modelDir = modelDir
        self.maxInferencePixels = maxInferencePixels
    }
}

public struct UNBGRemoveBackgroundResponse {
    public let modelUsed: String
    public let width: UInt32
    public let height: UInt32
    public let maskPng: Data
    public let providerSelected: String
    public let backendSelected: String?
    public let fallbackUsed: Bool
}

public enum UNBGClientError: Error {
    case invalidResponse
    case runtime(code: String, message: String)
}

public enum UNBGClient {
    public static func removeBackground(_ request: UNBGRemoveBackgroundRequest) throws -> UNBGRemoveBackgroundResponse {
        let requestPayload: [String: Any?] = [
            "image_bytes": request.imageBytes.map { Int($0) & 0xFF },
            "width": request.width,
            "height": request.height,
            "model": request.model,
            "onnx_variant": request.onnxVariant,
            "execution_provider": request.executionProvider,
            "gpu_backend": request.gpuBackend,
            "benchmark_provider": request.benchmarkProvider,
            "model_dir": request.modelDir,
            "max_inference_pixels": request.maxInferencePixels
        ]
        let requestData = try JSONSerialization.data(withJSONObject: requestPayload)
        guard let requestJson = String(data: requestData, encoding: .utf8) else {
            throw UNBGClientError.invalidResponse
        }

        let api = UnbgApi()
        let responseJson = api.removeBackgroundV1Json(requestJson: requestJson)
        guard let responseData = responseJson.data(using: .utf8),
              let root = try JSONSerialization.jsonObject(with: responseData) as? [String: Any] else {
            throw UNBGClientError.invalidResponse
        }

        if let code = root["code"] as? String {
            let message = root["message"] as? String ?? "unknown error"
            throw UNBGClientError.runtime(code: code, message: message)
        }

        guard let modelUsed = root["model_used"] as? String,
              let widthNumber = root["width"] as? NSNumber,
              let heightNumber = root["height"] as? NSNumber,
              let maskPng = root["mask_png"] as? [UInt8],
              let providerSelected = root["provider_selected"] as? String,
              let fallbackUsed = root["fallback_used"] as? Bool else {
            throw UNBGClientError.invalidResponse
        }

        return UNBGRemoveBackgroundResponse(
            modelUsed: modelUsed,
            width: widthNumber.uint32Value,
            height: heightNumber.uint32Value,
            maskPng: Data(maskPng),
            providerSelected: providerSelected,
            backendSelected: root["backend_selected"] as? String,
            fallbackUsed: fallbackUsed
        )
    }
}
