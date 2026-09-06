package com.example.novaai.api

import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*

data class CommandRequest(val text: String)
data class CommandResponse(val response: String, val action: String? = null)
data class VisionResponse(val response: String, val summary: List<Map<String, Any>>? = null)
data class OCRResponse(val response: String)
data class StatusResponse(val status: String, val name: String, val version: String)

interface NovaApiService {
    @GET("status")
    suspend fun getStatus(): StatusResponse

    @POST("command")
    suspend fun sendCommand(@Body request: CommandRequest): CommandResponse

    @Multipart
    @POST("vision")
    suspend fun uploadVision(@Part file: MultipartBody.Part): VisionResponse

    @Multipart
    @POST("ocr")
    suspend fun uploadOCR(@Part file: MultipartBody.Part): OCRResponse
}

object NovaApi {
    private const val BASE_URL = "http://10.220.167.194:8000/"

    private val logging = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    private val client = OkHttpClient.Builder()
        .addInterceptor(logging)
        .build()

    val service: NovaApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(NovaApiService::class.java)
    }
}
