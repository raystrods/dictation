# --- STEP 1: PATCH WITH GEVENT. This must be the first thing. ---
from gevent import monkey
monkey.patch_all()

# --- STEP 2: APPLY THE OFFICIAL GRPC GEVENT PATCH ---
import grpc.experimental.gevent
grpc.experimental.gevent.init_gevent()
# -----------------------------------------------------------------

# All other imports go AFTER the patches
import os
import queue
from flask import Flask, render_template
from flask_socketio import SocketIO
from google.cloud import speech

# Flask and SocketIO Setup
# NOTE: We must tell SocketIO we are using gevent as the async_mode.
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-very-secret-key'
socketio = SocketIO(app, async_mode='gevent')

# This queue will hold audio chunks sent from the browser.
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

@socketio.on('stop_stream')
def handle_stop_stream():
    """
    Called by the client when the 'Stop' button is clicked.
    This places a None sentinel into the queue to signal the end of the stream.
    """
    print('Client requested to stop stream. Sending None to queue.')
    audio_queue.put(None)

@socketio.on('start_stream')
def handle_start_stream(options):
    """
    Called by the client to start the transcription stream.
    This launches the main transcription logic in a background thread.
    """
    # This makes sure that if there's any leftover data from a
    # previous, abruptly-ended stream, it gets cleared out.
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            continue
    
    print(f"Starting transcription stream for client: {options['sid']}")
    socketio.start_background_task(target=transcribe_stream, client_sid=options['sid'])

@socketio.on('audio_stream')
def handle_audio_stream(audio_chunk):
    """
    Receives an audio chunk from the browser and puts it into our queue.
    """
    if not audio_queue.full():
        audio_queue.put(audio_chunk)


# --- Core Transcription Logic ---
def transcribe_stream(client_sid):
    """
    This is the core function that handles the gRPC stream to Google Cloud.
    It runs in a background thread for each client that starts a stream.
    """
    try:
        client = speech.SpeechClient()
        print(f"Google Cloud Speech client initialized successfully for SID: {client_sid}")
    except Exception as e:
        print(f"Error initializing Google Cloud Speech client for SID {client_sid}: {e}")
        socketio.emit('error', {'message': 'Server could not connect to speech service.'}, room=client_sid)
        return

    def stream_generator():
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,
        language_code="en-US",
        model="medical_dictation",
        enable_automatic_punctuation=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    print(f"Starting Google Cloud streaming recognize for SID: {client_sid}")
    requests = stream_generator()
    responses = client.streaming_recognize(config=streaming_config, requests=requests)

    try:
        for response in responses:
            if not response.results:
                continue
            result = response.results[0]
            if not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            is_final = result.is_final
            socketio.emit('transcript_update', {'transcript': transcript, 'is_final': is_final}, room=client_sid)
    except Exception as e:
        print(f"An error occurred during transcription for SID {client_sid}: {e}")
    finally:
        print(f"Transcription stream finished for SID: {client_sid}")
        socketio.emit('stream_finished', room=client_sid)


# --- Main Execution Block ---
if __name__ == '__main__':
    print("Starting Flask-SocketIO server with gevent...")
    socketio.run(app, debug=True)
