package com.unbg.sdk

import org.json.JSONArray
import org.json.JSONObject

/**
 * Thin Android-side facade over generated UniFFI bindings.
 * The generated bindings are expected under integrations/android-unbg/generated.
 */
object UnbgClient {
    data class RemoveBackgroundRequest(
        val imageBytes: ByteArray,
        val width: UInt,
        val height: UInt,
        val model: String = "auto",
        val onnxVariant: String? = "fp16",
        val executionProvider: String? = "auto",
        val gpuBackend: String? = "auto",
        val benchmarkProvider: Boolean? = true,
        val modelDir: String? = null,
        val maxInferencePixels: UInt? = 1_500_000u
    )

    data class RemoveBackgroundResponse(
        val modelUsed: String,
        val width: UInt,
        val height: UInt,
        val maskPng: ByteArray,
        val providerSelected: String,
        val backendSelected: String?,
        val fallbackUsed: Boolean
    )

    fun removeBackground(request: RemoveBackgroundRequest): RemoveBackgroundResponse {
        val requestJson = JSONObject()
            .put("image_bytes", JSONArray(request.imageBytes.map { it.toInt() and 0xFF }))
            .put("width", request.width.toLong())
            .put("height", request.height.toLong())
            .put("model", request.model)
            .put("onnx_variant", request.onnxVariant)
            .put("execution_provider", request.executionProvider)
            .put("gpu_backend", request.gpuBackend)
            .put("benchmark_provider", request.benchmarkProvider)
            .put("model_dir", request.modelDir)
            .put("max_inference_pixels", request.maxInferencePixels?.toLong())
            .toString()

        val api = unbg.UnbgApi()
        val responseJsonRaw = api.removeBackgroundV1Json(requestJson)
        val responseJson = JSONObject(responseJsonRaw)
        if (responseJson.has("code")) {
            throw IllegalStateException(
                "${responseJson.optString("code")}: ${responseJson.optString("message")}"
            )
        }

        return RemoveBackgroundResponse(
            modelUsed = responseJson.getString("model_used"),
            width = responseJson.getLong("width").toUInt(),
            height = responseJson.getLong("height").toUInt(),
            maskPng = jsonArrayToBytes(responseJson.getJSONArray("mask_png")),
            providerSelected = responseJson.getString("provider_selected"),
            backendSelected = responseJson.optString("backend_selected").ifEmpty { null },
            fallbackUsed = responseJson.getBoolean("fallback_used")
        )
    }

    private fun jsonArrayToBytes(array: JSONArray): ByteArray {
        val out = ByteArray(array.length())
        for (i in 0 until array.length()) {
            out[i] = (array.getInt(i) and 0xFF).toByte()
        }
        return out
    }
}
