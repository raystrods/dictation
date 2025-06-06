import queue

from flask import Flask, render_template
from flask_socketio import SocketIO
from google.cloud import speech

# --- Flask and SocketIO Setup ---
app = Flask(__name__)
# The secret key is used to sign the session cookie.
# It's important for security in a real application.
app.config['SECRET_KEY'] = 'your-very-secret-key'
socketio = SocketIO(app)

# --- Google Cloud Speech-to-Text Setup ---
client = speech.SpeechClient()
# This queue will hold audio chunks from the browser.
audio_queue = queue.Queue()

# --- Web App Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

# --- WebSocket Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """A new client has connected."""
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """A client has disconnected."""
    print('Client disconnected')
    # You might want to clean up resources here if a client disconnects mid-stream.

@socketio.on('start_stream')
def handle_start_stream(options):
    """Starts the transcription stream."""
    print('Starting transcription stream...')
    # Start the gRPC stream in a background thread.
    socketio.start_background_task(target=transcribe_stream, client_sid=options['sid'])

@socketio.on('audio_stream')
def handle_audio_stream(audio_chunk):
    """Receives an audio chunk from the browser and adds it to the queue."""
    audio_queue.put(audio_chunk)

# --- Core Transcription Logic ---
def transcribe_stream(client_sid):
    """The main gRPC streaming logic."""
    # The browser's MediaRecorder sends audio in a container format like WebM.
    # We need to configure the API to expect this format.
    # The sample rate is determined by the browser's recording settings.
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,  # Standard for web audio
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    # Create the generator for the gRPC stream
    def stream_generator():
        while True:
            # Get the audio chunk from the queue.
            # This will block until a chunk is available.
            chunk = audio_queue.get()
            if chunk is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    requests = stream_generator()
    responses = client.streaming_recognize(config=streaming_config, requests=requests)

    # Now, process the responses from the API
    try:
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            is_final = result.is_final

            # Send the transcript back to the specific client
            socketio.emit('transcript_update', {'transcript': transcript, 'is_final': is_final}, room=client_sid)

            if is_final:
                print(f"Final Transcript: {transcript}")
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Transcription stream finished.")
        # Signal the end of the stream
        socketio.emit('stream_finished', room=client_sid)


if __name__ == '__main__':
    print("Starting Flask server...")
    # Use socketio.run for development.
    # It automatically uses the best available asynchronous server (like eventlet).
    socketio.run(app, debug=True)
