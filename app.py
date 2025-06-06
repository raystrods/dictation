# --- THIS MUST BE THE VERY FIRST THING IN YOUR APP ---
import eventlet
eventlet.monkey_patch()

import os
import queue

from flask import Flask, render_template
from flask_socketio import SocketIO
from google.cloud import speech

# --- Flask and SocketIO Setup ---
# This initializes the Flask app.
app = Flask(__name__)
# The secret key is used by Flask to sign the session cookie.
# It's a good practice for security, though less critical for this specific app's functionality.
app.config['SECRET_KEY'] = 'your-very-secret-key'
socketio = SocketIO(app)


# --- Google Cloud Speech-to-Text Setup ---
# This instantiates the Speech-to-Text client.
# IMPORTANT: By leaving the parentheses empty, the client library will automatically
# look for the GOOGLE_APPLICATION_CREDENTIALS environment variable. On Render, you
# will set this to point to your secret JSON file. This is the secure, recommended
# way to handle authentication on a server.
try:
    client = speech.SpeechClient()
    print("Google Cloud Speech client initialized successfully.")
except Exception as e:
    print(f"Error initializing Google Cloud Speech client: {e}")
    print("Please ensure the GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly.")
    client = None

# This queue will hold audio chunks sent from the browser.
# It acts as a bridge between the WebSocket handler and the gRPC background task.
audio_queue = queue.Queue()


# --- Web App Routes ---
@app.route('/')
def index():
    """Serves the main HTML page for the web application."""
    return render_template('index.html')


# --- WebSocket Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """A new client has connected to the WebSocket."""
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """A client has disconnected from the WebSocket."""
    print('Client disconnected')

@socketio.on('start_stream')
def handle_start_stream(options):
    """
    Called by the client to start the transcription stream.
    This launches the main transcription logic in a background thread.
    """
    if client is None:
        socketio.emit('error', {'message': 'Server is not configured with API credentials.'}, room=options['sid'])
        return
        
    print(f"Starting transcription stream for client: {options['sid']}")
    # The `socketio.start_background_task` is essential for running a long-lived
    # function like our gRPC stream without blocking the main web server.
    socketio.start_background_task(target=transcribe_stream, client_sid=options['sid'])

@socketio.on('audio_stream')
def handle_audio_stream(audio_chunk):
    """
    Receives an audio chunk from the browser and puts it into our queue.
    """
    audio_queue.put(audio_chunk)


# --- Core Transcription Logic ---
def transcribe_stream(client_sid):
    """
    This is the core function that handles the gRPC stream to Google Cloud.
    It runs in a background thread for each client that starts a stream.
    """
    # This is the generator function that pulls audio chunks from the queue.
    def stream_generator():
        while True:
            # The `audio_queue.get()` call will block until an item is available.
            chunk = audio_queue.get()
            # A `None` item in the queue is our signal to end the stream.
            if chunk is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    # The RecognitionConfig object defines the audio format and model.
    # The browser's MediaRecorder using 'audio/webm;codecs=opus' is the source.
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,  # Standard for web audio recording
        language_code="en-US",
        # This is the key line to enable the specialized medical model.
        model="medical_dictation",
        enable_automatic_punctuation=True,
    )

    # The StreamingRecognitionConfig object configures the streaming behavior.
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,  # Get intermediate results for a real-time feel
    )

    print(f"Starting Google Cloud streaming recognize for SID: {client_sid}")
    requests = stream_generator()
    responses = client.streaming_recognize(config=streaming_config, requests=requests)

    # Now, process the responses from the API and send them back to the client.
    try:
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            is_final = result.is_final

            # Emit the transcript update back to the specific client's browser.
            socketio.emit('transcript_update', {'transcript': transcript, 'is_final': is_final}, room=client_sid)

    except Exception as e:
        print(f"An error occurred during transcription for SID {client_sid}: {e}")
    finally:
        print(f"Transcription stream finished for SID: {client_sid}")
        # Let the client know the stream has officially ended.
        socketio.emit('stream_finished', room=client_sid)


# --- Main Execution Block ---
if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    # `socketio.run` is the correct way to start the server.
    # It automatically uses the best available asynchronous server (like eventlet).
    # `debug=True` enables auto-reloading when you save changes to the file.
    socketio.run(app, debug=True)
