package com.example.mentalhealthchat

import android.content.Intent
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.view.Gravity
import android.view.View
import android.view.animation.AnimationUtils
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.android.material.switchmaterial.SwitchMaterial
import okhttp3.*
import org.json.JSONObject
import java.io.IOException
import java.util.*
import java.util.concurrent.TimeUnit
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import android.content.pm.PackageManager
import androidx.core.app.ActivityCompat


class MainActivity : AppCompatActivity() {
    private lateinit var chatHistory: LinearLayout
    private lateinit var scrollView: ScrollView
    private lateinit var inputMessage: EditText
    private lateinit var sendButton: Button
    private lateinit var typingIndicator: TextView
    private lateinit var responseToggle: SwitchMaterial
    private lateinit var suggestionChips: ChipGroup
    private lateinit var micButton: ImageButton
    private lateinit var resetChatButton: Button
    private lateinit var listeningIndicator: TextView

    private val speechRecognizer: SpeechRecognizer by lazy {
        SpeechRecognizer.createSpeechRecognizer(this)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Microphone permission granted", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "Microphone permission denied", Toast.LENGTH_SHORT).show()
        }
    }


    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    private val apiUrl = "Replace_with_your_endpoint_url/chat" // Replace with your endpoint

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        //Request microphone permission
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.RECORD_AUDIO), 1)
        }

        //View bindings
        chatHistory = findViewById(R.id.chatHistory)
        scrollView = findViewById(R.id.scrollView)
        inputMessage = findViewById(R.id.inputMessage)
        sendButton = findViewById(R.id.sendButton)
        typingIndicator = findViewById(R.id.typingIndicator)
        responseToggle = findViewById(R.id.responseToggle)
        suggestionChips = findViewById(R.id.suggestionChips)
        micButton = findViewById(R.id.micButton)
        resetChatButton = findViewById(R.id.resetChatButton)
        listeningIndicator = findViewById(R.id.listeningIndicator)

        setupSuggestionChips()
        setupVoiceInput()

        sendButton.setOnClickListener {
            val userMessage = inputMessage.text.toString().trim()
            if (userMessage.isNotEmpty()) sendUserMessage(userMessage)
        }

        resetChatButton.setOnClickListener {
            chatHistory.removeAllViews()
            suggestionChips.removeAllViews()
            setupSuggestionChips()
            suggestionChips.visibility = View.VISIBLE
        }

        micButton.setOnClickListener {
            startVoiceRecognition()
        }
    }


    private fun setupSuggestionChips() {
        val suggestions = listOf(
            "Why do I overthink everything?",
            "I'm feeling down lately.",
            "How to improve my sleep?"
        )
        suggestions.forEach { suggestionText ->
            val chip = Chip(this).apply {
                text = suggestionText
                isClickable = true
                isCheckable = false
                setChipBackgroundColorResource(R.color.transparent_chip_bg)
                setTextColor(ContextCompat.getColor(this@MainActivity, android.R.color.black))
                textSize = 13f
            }
            chip.setOnClickListener {
                sendUserMessage(suggestionText)
            }
            suggestionChips.addView(chip)
        }
    }

    private fun setupVoiceInput() {
        speechRecognizer.setRecognitionListener(object : android.speech.RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                listeningIndicator.visibility = View.VISIBLE
            }

            override fun onResults(results: Bundle?) {
                listeningIndicator.visibility = View.GONE
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                matches?.firstOrNull()?.let { sendUserMessage(it) }
            }

            override fun onError(error: Int) {
                listeningIndicator.visibility = View.GONE
                Toast.makeText(this@MainActivity, "Voice input failed", Toast.LENGTH_SHORT).show()
            }

            override fun onBeginningOfSpeech() {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onRmsChanged(rmsdB: Float) {}
        })
    }

    private fun startVoiceRecognition() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
        }
        speechRecognizer.startListening(intent)
    }

    private fun sendUserMessage(message: String) {
        suggestionChips.visibility = View.GONE
        addChatBubble(message, isUser = true)
        inputMessage.setText("")
        showTyping(true)
        sendMessageToBot(message)
    }

    private fun sendMessageToBot(message: String) {
        val json = JSONObject().apply {
            put("message", message)
            put("mode", if (responseToggle.isChecked) "long" else "short")
        }

        val requestBody = json.toString().toRequestBody("application/json".toMediaType())
        val request = Request.Builder().url(apiUrl).post(requestBody).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    showTyping(false)
                    addChatBubble("❌ Failed: ${e.message}", isUser = false)
                }
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    val body = response.body?.string()
                    val botReply = JSONObject(body ?: "{}").optString("response", "❌ No response")
                    runOnUiThread {
                        showTyping(false)
                        addChatBubble("🤖: $botReply", isUser = false)
                    }
                }
            }
        })
    }

    private fun addChatBubble(message: String, isUser: Boolean) {
        val bubbleLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(12, 6, 12, 6) }
        }

        val avatar = ImageView(this).apply {
            setImageResource(if (isUser) R.drawable.user else R.drawable.bot)
            layoutParams = LinearLayout.LayoutParams(60, 60)
        }

        val textView = TextView(this).apply {
            text = message
            textSize = 16f
            setTextColor(if (isUser) 0xFFFFFFFF.toInt() else 0xFF000000.toInt())
            setPadding(16, 16, 16, 16)
            background = getDrawable(if (isUser) R.drawable.user_bubble else R.drawable.bot_bubble)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                setMargins(8, 4, 8, 4)
            }
            maxWidth = (resources.displayMetrics.widthPixels * 0.75).toInt()  // Limit to 75% of screen
        }


        val innerLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = if (isUser) Gravity.END else Gravity.START
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { gravity = if (isUser) Gravity.END else Gravity.START }
        }

        if (isUser) {
            innerLayout.addView(textView)
            innerLayout.addView(avatar)
            bubbleLayout.gravity = Gravity.END
        } else {
            innerLayout.addView(avatar)
            innerLayout.addView(textView)
            bubbleLayout.gravity = Gravity.START
        }

        bubbleLayout.addView(innerLayout)
        bubbleLayout.startAnimation(AnimationUtils.loadAnimation(this, R.anim.fade_in))
        chatHistory.addView(bubbleLayout)

        scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
    }

    private fun showTyping(show: Boolean) {
        typingIndicator.visibility = if (show) View.VISIBLE else View.GONE
    }
}
