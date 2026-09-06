package com.example.novaai.ui.main

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.novaai.api.CommandRequest
import com.example.novaai.api.NovaApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

data class Message(val text: String, val isUser: Boolean)

class MainScreenViewModel : ViewModel() {
    private val _messages = MutableStateFlow<List<Message>>(listOf(Message("Hello Captain, I am Nova AI. How can I help you?", false)))
    val messages: StateFlow<List<Message>> = _messages.asStateFlow()

    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()

    init {
        checkStatus()
    }

    fun checkStatus() {
        viewModelScope.launch {
            try {
                val status = NovaApi.service.getStatus()
                addSystemMessage("Connected to Nova Brain v${status.version}")
            } catch (e: Exception) {
                addSystemMessage("Connecting to brain... (Check if server is running on PC)")
            }
        }
    }

    fun sendCommand(text: String) {
        if (text.isBlank()) return
        
        viewModelScope.launch {
            _messages.value += Message(text, true)
            _isProcessing.value = true
            
            try {
                val response = NovaApi.service.sendCommand(CommandRequest(text))
                _messages.value += Message(response.response, false)
            } catch (e: Exception) {
                _messages.value += Message("Error: ${e.localizedMessage}", false)
            } finally {
                _isProcessing.value = false
            }
        }
    }

    fun analyzeImage(imageBytes: ByteArray) {
        viewModelScope.launch {
            _messages.value += Message("[Image Sent for Analysis]", true)
            _isProcessing.value = true
            
            try {
                val requestBody = imageBytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
                val body = MultipartBody.Part.createFormData("file", "image.jpg", requestBody)
                val response = NovaApi.service.uploadVision(body)
                _messages.value += Message(response.response, false)
            } catch (e: Exception) {
                _messages.value += Message("Vision Error: ${e.localizedMessage}", false)
            } finally {
                _isProcessing.value = false
            }
        }
    }
    
    fun addSystemMessage(text: String) {
        _messages.value += Message(text, false)
    }
}
