import sys
import os
import datetime
import random
import numpy as np
import torch
import tempfile
import gc
import warnings

# Suppress specific warnings from the ChatterBox library
warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*LlamaSdpaAttention.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*past_key_values.*tuple of tuples.*", category=FutureWarning)

# Set environment variable to use eager attention implementation to avoid fallback warning
os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"

# Enhanced device detection with optimization flags
def get_optimal_device():
    if torch.cuda.is_available():
        device = 'cuda'
        # Enable TF32 on Ampere GPUs for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        # MPS-specific optimizations
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Better memory management
    else:
        device = 'cpu'
        # Enable CPU optimizations
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    
    return device

map_location = get_optimal_device()

# Context manager for torch.load patching instead of global patching
class PatchedTorchLoad:
    def __enter__(self):
        self.original = torch.load
        def patched_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = map_location
            return self.original(*args, **kwargs)
        torch.load = patched_load
        return self
    
    def __exit__(self, *args):
        torch.load = self.original

# Use context manager when needed
with PatchedTorchLoad():
    pass  # Will be used during model loading
import torchaudio
import traceback

# --- NLTK Import and Setup (Same as your working version) ---
try:
    import nltk
    NLTK_RESOURCES_OK = True
    nltk_resources_to_check = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab"
    }
    for resource_id, resource_path in nltk_resources_to_check.items():
        try:
            nltk.data.find(resource_path)
            print(
                f"NLTK '{resource_id}' (path: {resource_path}) resource found.")
        except LookupError:
            print(
                f"NLTK '{resource_id}' (path: {resource_path}) resource not found. Attempting to download '{resource_id}'...")
            try:
                nltk.download(resource_id, quiet=False)
                nltk.data.find(resource_path)
                print(
                    f"NLTK '{resource_id}' resource downloaded and verified successfully.")
            except Exception as e_download:
                print(
                    f"ERROR: Failed to download or verify NLTK '{resource_id}' resource: {e_download}")
                print(f"  python -m nltk.downloader {resource_id}")
                NLTK_RESOURCES_OK = False
                break
        except Exception as e_other:
            print(
                f"ERROR: Unexpected error while checking NLTK '{resource_id}' resource: {e_other}")
            NLTK_RESOURCES_OK = False
            break
except ImportError:
    print("FATAL ERROR: NLTK library not found.")
    nltk = None
    NLTK_RESOURCES_OK = False
# --- END NLTK ---

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QTextEdit, QPushButton, QSlider, QSpinBox,
    QFileDialog, QMessageBox, QListWidget, QListWidgetItem, QGroupBox,
    QCheckBox
)
# QStandardPaths was in your full file, good.
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer, QTime
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaDevices

try:
    from chatterbox.tts import ChatterboxTTS
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    ChatterboxTTS = None
    print("WARNING: chatterbox-tts library not found.")

MAX_TEXT_INPUT_LENGTH = 280
EFFECTIVE_MAX_CHUNK_LENGTH = MAX_TEXT_INPUT_LENGTH - 20

# --- ModelLoaderThread (Same as your working version) ---


class ModelLoaderThread(QThread):
    model_loaded = Signal(object, str)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.device = map_location

    def run(self):
        try:
            if not CHATTERBOX_AVAILABLE:
                self.error_occurred.emit(
                    "ChatterboxTTS library is not installed.")
                return
            print(
                f"Attempting to load ChatterboxTTS model on device: {self.device}...")
            
            # Load model with context manager for proper device mapping
            with PatchedTorchLoad():
                # Try to pass attention implementation if supported
                try:
                    model_instance = ChatterboxTTS.from_pretrained(
                        self.device, 
                        attn_implementation="eager"
                    )
                except TypeError:
                    # Fallback if attn_implementation is not supported
                    model_instance = ChatterboxTTS.from_pretrained(self.device)
            
            print(f"Model type: {type(model_instance)}")
            print(f"Model methods: {[m for m in dir(model_instance) if not m.startswith('_')][:10]}...")  # First 10 public methods
            
            if hasattr(model_instance, 'to'):
                model_instance.to(self.device)
            
            # Apply optimizations based on device
            if self.device == 'mps':
                # MPS-specific optimizations
                if hasattr(model_instance, 'eval'):
                    model_instance.eval()  # Ensure eval mode
                    print("Model set to eval mode")
                else:
                    print("Model does not have eval() method")
                # Enable mixed precision for MPS if supported
                if hasattr(torch, 'autocast') and hasattr(torch, 'mps'):
                    print("Enabling MPS optimizations...")
            elif self.device == 'cuda':
                if hasattr(model_instance, 'eval'):
                    model_instance.eval()
                    print("Model set to eval mode")
                else:
                    print("Model does not have eval() method")
                # Try to compile model if PyTorch 2.0+
                if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
                    try:
                        print("Attempting to compile model with torch.compile()...")
                        model_instance = torch.compile(model_instance, mode='reduce-overhead')
                        print("Model compiled successfully!")
                    except Exception as e:
                        print(f"torch.compile() failed (will use uncompiled): {e}")
            
            print(
                f"Model loaded successfully. Model device: {model_instance.device if hasattr(model_instance, 'device') else 'N/A'}")
            self.model_loaded.emit(model_instance, self.device)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error loading model: {e}\nTraceback:\n{tb_str}")
            self.error_occurred.emit(
                f"Failed to load model: {str(e)}\nSee console for traceback.")

# --- AudioGeneratorThread (Same as your working version with stop flag) ---


class AudioGeneratorThread(QThread):
    generation_complete = Signal(str, int)
    error_occurred = Signal(str)
    chunk_generated = Signal(int, int)

    def __init__(self, model, text, audio_prompt_path, exaggeration, temperature, cfg_weight, seed, output_dir):
        super().__init__()
        self.model = model
        self.original_text = text
        self.audio_prompt_path = audio_prompt_path
        self.exaggeration = exaggeration
        self.temperature = temperature
        self.cfg_weight = cfg_weight
        self.input_seed = seed
        self.output_dir = output_dir
        self.actual_seed_used = seed
        self._is_stopped = False
        self.device = map_location  # Add device detection
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def stop(self):
        print("Stop requested for audio generation thread.")
        self._is_stopped = True
    
    def _save_audio_with_platform_handling(self, audio_tensor, output_path, sample_rate):
        """Save audio with platform-specific format handling"""
        if sys.platform == "darwin":  # macOS specific handling
            # macOS Qt audio player has issues with 24kHz float32 audio
            # Convert to 16-bit PCM and resample to 48kHz for compatibility
            
            # Normalize audio
            max_val = torch.max(torch.abs(audio_tensor))
            if max_val > 0:
                audio_normalized = audio_tensor / max_val
            else:
                print("WARNING: Audio appears to be silent (max amplitude is 0)")
                audio_normalized = audio_tensor
            
            # Resample to 48kHz if needed
            if sample_rate != 48000:
                import torchaudio.transforms as T
                resampler = T.Resample(sample_rate, 48000)
                audio_resampled = resampler(audio_normalized)
                audio_int16 = (audio_resampled * 32767).to(torch.int16)
                torchaudio.save(output_path, audio_int16, 48000)
                print(f"Final audio saved (16-bit PCM, 48kHz) to: {output_path}")
            else:
                audio_int16 = (audio_normalized * 32767).to(torch.int16)
                torchaudio.save(output_path, audio_int16, sample_rate)
                print(f"Final audio saved (16-bit PCM) to: {output_path}")
        else:
            # Standard save for other platforms
            torchaudio.save(output_path, audio_tensor, sample_rate)
            print(f"Final stitched audio saved to: {output_path}")

    def set_seed_internal(self, seed_val: int):
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)  # Corrected from manual_seed
        random.seed(seed_val)
        np.random.seed(seed_val)
        print(f"Seed set to: {seed_val}")
        self.actual_seed_used = seed_val

    def _chunk_long_sentence(self, sentence, max_len):
        sub_chunks = []
        current_pos = 0
        sentence_len = len(sentence)
        while current_pos < sentence_len:
            end_pos = min(current_pos + max_len, sentence_len)
            if end_pos == sentence_len:
                sub_chunks.append(sentence[current_pos:end_pos].strip())
                current_pos = end_pos
            else:
                last_space_idx = sentence.rfind(' ', current_pos, end_pos)
                if last_space_idx != -1 and last_space_idx > current_pos:
                    sub_chunks.append(
                        sentence[current_pos:last_space_idx].strip())
                    current_pos = last_space_idx + 1
                else:
                    sub_chunks.append(sentence[current_pos:end_pos].strip())
                    current_pos = end_pos
        return [sc for sc in sub_chunks if sc]

    def run(self):
        try:
            if nltk is None or not NLTK_RESOURCES_OK:
                self.error_occurred.emit(
                    "NLTK or 'punkt' missing. Check setup.")
                return
            if self.model is None:
                self.error_occurred.emit("Model not loaded.")
                return
            if self._is_stopped:
                self.error_occurred.emit("Stopped by user before start.")
                return

            if self.input_seed == 0:
                r_seed = random.randint(1, 1_000_000)
                self.set_seed_internal(r_seed)
                print(
                    f"Input seed 0. Using random seed: {self.actual_seed_used}")
            else:
                self.set_seed_internal(self.input_seed)

            final_chunks = []
            text_to_process = self.original_text.strip()
            print("Using NLTK for sentence tokenization/combining...")
            sentences = nltk.sent_tokenize(text_to_process)
            current_chunk_sents = []
            current_chunk_len = 0
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) > MAX_TEXT_INPUT_LENGTH:
                    if current_chunk_sents:
                        final_chunks.append(" ".join(current_chunk_sents))
                    current_chunk_sents = []
                    current_chunk_len = 0
                    print(
                        f"Sentence too long ({len(sentence)}), sub-chunking: \"{sentence[:30]}...\"")
                    final_chunks.extend(self._chunk_long_sentence(
                        sentence, EFFECTIVE_MAX_CHUNK_LENGTH))
                    continue
                potential_len = current_chunk_len + \
                    (1 if current_chunk_sents else 0) + len(sentence)
                if potential_len <= MAX_TEXT_INPUT_LENGTH:
                    current_chunk_sents.append(sentence)
                    current_chunk_len = potential_len
                else:
                    if current_chunk_sents:
                        final_chunks.append(" ".join(current_chunk_sents))
                    current_chunk_sents = [sentence]
                    current_chunk_len = len(sentence)
            if current_chunk_sents:
                final_chunks.append(" ".join(current_chunk_sents))
            final_chunks = [c.strip() for c in final_chunks if c.strip()]

            if not final_chunks:
                self.error_occurred.emit(
                    "Input text empty or resulted in no chunks.")
                return

            total_chunks = len(final_chunks)
            print(f"Processed into {total_chunks} chunks.")
            # (Optional debug print for chunks can go here)

            # Streaming audio generation with better memory management
            sr = self.model.sr
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chatterbox_{timestamp}_seed{self.actual_seed_used}_full_stitched.wav"
            output_path = os.path.join(self.output_dir, filename)
            
            # Create temporary file for streaming chunks
            temp_dir = tempfile.gettempdir()
            temp_chunks = []
            
            try:
                # Generate chunks and save to temporary files
                for i, chunk_text in enumerate(final_chunks):
                    if self._is_stopped:
                        self.error_occurred.emit(
                            f"Generation stopped by user at chunk {i+1}/{total_chunks}.")
                        return
                    
                    current_chunk_num = i + 1
                    self.chunk_generated.emit(current_chunk_num, total_chunks)
                    print(
                        f"\nGenerating chunk {current_chunk_num}/{total_chunks} (seed: {self.actual_seed_used}): '{chunk_text[:70]}...'")
                    
                    # Use autocast for MPS/CUDA if available
                    if self.device == 'mps' and hasattr(torch, 'autocast'):
                        with torch.autocast('mps', dtype=torch.float16):
                            wav_tensor_chunk = self.model.generate(chunk_text,
                                                                 audio_prompt_path=self.audio_prompt_path if self.audio_prompt_path else None,
                                                                 exaggeration=self.exaggeration, 
                                                                 temperature=self.temperature, 
                                                                 cfg_weight=self.cfg_weight)
                    elif self.device == 'cuda':
                        with torch.cuda.amp.autocast(enabled=True):
                            wav_tensor_chunk = self.model.generate(chunk_text,
                                                                 audio_prompt_path=self.audio_prompt_path if self.audio_prompt_path else None,
                                                                 exaggeration=self.exaggeration, 
                                                                 temperature=self.temperature, 
                                                                 cfg_weight=self.cfg_weight)
                    else:
                        wav_tensor_chunk = self.model.generate(chunk_text,
                                                             audio_prompt_path=self.audio_prompt_path if self.audio_prompt_path else None,
                                                             exaggeration=self.exaggeration, 
                                                             temperature=self.temperature, 
                                                             cfg_weight=self.cfg_weight)
                    
                    if wav_tensor_chunk.ndim == 1:
                        wav_tensor_chunk = wav_tensor_chunk.unsqueeze(0)
                    
                    # Save chunk to temporary file immediately
                    temp_chunk_path = os.path.join(temp_dir, f"chunk_{i:04d}.wav")
                    torchaudio.save(temp_chunk_path, wav_tensor_chunk.cpu(), sr)
                    temp_chunks.append(temp_chunk_path)
                    
                    # Clear GPU memory after each chunk
                    del wav_tensor_chunk
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    elif self.device == 'mps':
                        torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
                    gc.collect()
                
                if self._is_stopped:
                    self.error_occurred.emit("Stopped before final concat.")
                    return
                
                if not temp_chunks:
                    self.error_occurred.emit("No audio data generated.")
                    return
                
                print("\nConcatenating audio chunks...")
                
                # Efficient concatenation using temporary files
                audio_chunks = []
                for chunk_path in temp_chunks:
                    chunk_audio, _ = torchaudio.load(chunk_path)
                    audio_chunks.append(chunk_audio)
                
                # Concatenate in smaller batches to avoid memory spike
                if len(audio_chunks) > 10:
                    final_audio = audio_chunks[0]
                    for chunk in audio_chunks[1:]:
                        final_audio = torch.cat([final_audio, chunk], dim=1)
                        # Clear intermediate tensors
                        del chunk
                        gc.collect()
                else:
                    final_audio = torch.cat(audio_chunks, dim=1)
                
                # Save final audio with platform-specific handling
                self._save_audio_with_platform_handling(final_audio, output_path, sr)
                
                # Cleanup
                del final_audio
                del audio_chunks
                gc.collect()
                
            finally:
                # Clean up temporary files
                for temp_file in temp_chunks:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except:
                        pass
            
            self.generation_complete.emit(output_path, sr)
        except Exception as e:
            if not self._is_stopped:
                tb_str = traceback.format_exc()
                print(f"Error in AudioGeneratorThread: {e}\n{tb_str}")
                self.error_occurred.emit(
                    f"Generation/stitching error: {str(e)}")

# --- ChatterboxApp ---


class ChatterboxApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chatterbox TTS Interface")
        self.setGeometry(100, 100, 800, 720)
        self.model = None
        self.device_used = "cpu"
        self.current_audio_file = None
        self.output_directory = os.path.join(os.getcwd(), "chatterbox_outputs")
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.last_reference_audio_dir = script_dir

        self.media_player = QMediaPlayer()
        
        # Initialize audio output with platform-specific handling
        self._init_audio_output()
        
        self.media_player.setAudioOutput(self.audio_output)
        self.is_seeking_audio = False
        self.paused_position = 0  # Added paused_position here

        self.media_player.positionChanged.connect(self.update_slider_position)
        self.media_player.durationChanged.connect(self.update_duration_info)
        self.media_player.playbackStateChanged.connect(
            self.handle_playback_state_changed)
        self.media_player.errorOccurred.connect(self.handle_media_error)
        self.media_player.mediaStatusChanged.connect(self.handle_media_status_changed)

        self.generation_timer = QTimer(self)
        self.generation_timer.timeout.connect(
            self.update_generation_time_display)  # Renamed for clarity
        self.generation_start_time = None
        self.is_generating = False

        self._init_ui()
        self.update_output_log()
        if CHATTERBOX_AVAILABLE:
            self.load_model()
        else:
            self.status_bar.setText("Status: Chatterbox library not found.")
            self.generate_button.setEnabled(False)
            self.load_model_button.setEnabled(False)

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        inputs_group = QGroupBox("Inputs")
        form_layout = QFormLayout()

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            f"Enter text to synthesize (approx. {MAX_TEXT_INPUT_LENGTH} chars recommended).")
        self.text_input.setFixedHeight(100)
        form_layout.addRow(QLabel("Text:"), self.text_input)

        ref_audio_layout = QHBoxLayout()
        self.ref_audio_path_label = QLabel("None selected.")
        self.ref_audio_path_label.setWordWrap(True)
        browse_ref_button = QPushButton("Browse Reference Audio...")
        browse_ref_button.clicked.connect(self.browse_reference_audio)
        ref_audio_layout.addWidget(self.ref_audio_path_label, 1)
        ref_audio_layout.addWidget(browse_ref_button)
        form_layout.addRow(QLabel("Reference Audio:"), ref_audio_layout)

        self.exaggeration_slider = self._create_slider(0.25, 2.0, 0.05, 0.5)
        form_layout.addRow(
            QLabel("Exaggeration (0.25 - 2.0):"), self.exaggeration_slider)

        self.cfg_slider = self._create_slider(0.2, 1.0, 0.05, 0.5)
        form_layout.addRow(QLabel("CFG/Pace (0.2 - 1.0):"), self.cfg_slider)

        self.temp_slider = self._create_slider(0.05, 5.0, 0.05, 0.8)
        form_layout.addRow(
            QLabel("Temperature (0.05 - 5.0):"), self.temp_slider)

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 1_000_000_000)
        self.seed_input.setValue(0)
        form_layout.addRow(QLabel("Seed (0 for random):"), self.seed_input)

        inputs_group.setLayout(form_layout)
        main_layout.addWidget(inputs_group)

        controls_options_layout = QHBoxLayout()

        self.load_model_button = QPushButton("Reload Model")
        self.load_model_button.clicked.connect(self.load_model)
        controls_options_layout.addWidget(self.load_model_button)

        self.generate_button = QPushButton("Generate Audio")
        self.generate_button.clicked.connect(self.handle_generate_stop_toggle)
        self.generate_button.setEnabled(False)
        controls_options_layout.addWidget(self.generate_button)

        self.autoplay_checkbox = QCheckBox("Auto-play generated audio")
        self.autoplay_checkbox.setChecked(True)
        controls_options_layout.addWidget(self.autoplay_checkbox)
        controls_options_layout.addStretch()

        main_layout.addLayout(controls_options_layout)

        playback_group = QGroupBox("Playback & Output")
        playback_v_layout = QVBoxLayout()
        self.current_file_label = QLabel("Currently playing: None")
        playback_v_layout.addWidget(self.current_file_label)

        player_controls_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(
            self.toggle_play_pause)  # Connection is correct
        self.play_pause_button.setEnabled(False)
        player_controls_layout.addWidget(self.play_pause_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        self.stop_button.setEnabled(False)
        player_controls_layout.addWidget(self.stop_button)
        
        # Add volume control
        player_controls_layout.addWidget(QLabel("Volume:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.valueChanged.connect(self.update_volume)
        player_controls_layout.addWidget(self.volume_slider)
        
        self.volume_label = QLabel("100%")
        player_controls_layout.addWidget(self.volume_label)
        
        # Add test sound button for debugging
        self.test_sound_button = QPushButton("Test Sound")
        self.test_sound_button.clicked.connect(self.play_test_sound)
        player_controls_layout.addWidget(self.test_sound_button)
        
        playback_v_layout.addLayout(player_controls_layout)

        playhead_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.playhead_slider = QSlider(Qt.Horizontal)

        self.playhead_slider.sliderPressed.connect(self.slider_pressed)
        self.playhead_slider.sliderMoved.connect(self.seek_audio_on_move)
        self.playhead_slider.sliderReleased.connect(self.slider_released)

        self.playhead_slider.setEnabled(False)
        self.duration_label = QLabel("00:00")
        playhead_layout.addWidget(self.current_time_label)
        playhead_layout.addWidget(self.playhead_slider)
        playhead_layout.addWidget(self.duration_label)
        playback_v_layout.addLayout(playhead_layout)

        playback_v_layout.addWidget(
            QLabel("Generated Files History (double-click to play):"))
        self.output_log_listwidget = QListWidget()
        self.output_log_listwidget.itemDoubleClicked.connect(
            self.play_selected_from_log)
        playback_v_layout.addWidget(self.output_log_listwidget)

        playback_group.setLayout(playback_v_layout)
        main_layout.addWidget(playback_group)

        self.status_bar = QLabel("Status: Initializing...")
        main_layout.addWidget(self.status_bar)

    def handle_generate_stop_toggle(self):
        if not self.is_generating:
            # --- Start Generation Part ---
            # (Same as your last full working version, ensures button is enabled for stop)
            if self.model is None:
                QMessageBox.warning(self, "Model Not Loaded",
                                    "Please load the model first.")
                return
            text = self.text_input.toPlainText().strip()
            if not text:
                QMessageBox.warning(self, "Input Error",
                                    "Please enter some text to synthesize.")
                return

            self.is_generating = True
            self.generate_button.setText("Stop Generation")
            # Keep enabled to click "Stop"
            self.generate_button.setEnabled(True)
            self.load_model_button.setEnabled(False)

            self.generation_start_time = QTime.currentTime()
            self.generation_timer.start(1000)
            self.update_generation_time_display()  # Initial status update

            # ... (rest of parameter fetching and thread creation/start same as your file)
            ref_audio_full_path = self.ref_audio_path_label.toolTip()
            exaggeration = self.exaggeration_slider.get_value()
            cfg = self.cfg_slider.get_value()
            temperature = self.temp_slider.get_value()
            seed = self.seed_input.value()

            self.audio_generator_thread = AudioGeneratorThread(
                self.model, text, ref_audio_full_path, exaggeration, temperature, cfg, seed, self.output_directory
            )
            self.audio_generator_thread.generation_complete.connect(
                self.on_generation_complete)
            self.audio_generator_thread.error_occurred.connect(
                self.on_generation_error)
            self.audio_generator_thread.chunk_generated.connect(
                self.on_chunk_generated_progress)
            self.audio_generator_thread.finished.connect(
                self.on_generation_thread_finished)
            self.audio_generator_thread.start()

        else:  # self.is_generating is True, so this is a Stop request
            if hasattr(self, 'audio_generator_thread') and self.audio_generator_thread.isRunning():
                print("UI: Requesting stop for audio_generator_thread")
                self.audio_generator_thread.stop()  # Signal the thread
                self.generate_button.setText("Stopping...")
                # Disable button while waiting for thread to acknowledge stop
                self.generate_button.setEnabled(False)
                self.status_bar.setText(
                    "Status: Stop requested. Waiting for current chunk to finish...")
            else:  # Should not happen if is_generating is True
                print(
                    "UI: Stop requested, but no active generation thread found. Resetting UI.")
                self.on_generation_thread_finished()  # Manually trigger UI reset

    def _create_slider(self, min_val, max_val, step_val, default_val):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val/step_val))
        slider.setMaximum(int(max_val/step_val))
        slider.setValue(int(default_val/step_val))
        slider.setSingleStep(1)
        value_label = QLabel(f"{default_val:.2f}")
        slider.valueChanged.connect(
            lambda val, lbl=value_label, s=step_val: lbl.setText(f"{val*s:.2f}"))
        layout.addWidget(slider)
        layout.addWidget(value_label)
        container.get_value = lambda s=slider, st=step_val: s.value()*st
        return container

    def browse_reference_audio(self):
        default_dir = self.last_reference_audio_dir
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", default_dir, "Audio Files (*.wav *.mp3 *.flac)")
        if file_path:
            self.ref_audio_path_label.setText(os.path.basename(file_path))
            self.ref_audio_path_label.setToolTip(file_path)
            self.last_reference_audio_dir = os.path.dirname(file_path)
        else:
            self.ref_audio_path_label.setText("None selected.")
            self.ref_audio_path_label.setToolTip("")

    def load_model(self):
        if not CHATTERBOX_AVAILABLE:
            QMessageBox.critical(
                self, "Error", "ChatterboxTTS library not installed.")
            return
        self.status_bar.setText("Status: Loading model...")
        self.generate_button.setEnabled(False)
        self.load_model_button.setEnabled(False)
        self.model_loader_thread = ModelLoaderThread()
        self.model_loader_thread.model_loaded.connect(self.on_model_loaded)
        self.model_loader_thread.error_occurred.connect(
            self.on_model_load_error)
        self.model_loader_thread.start()

    def on_model_loaded(self, model_instance, device_used):
        self.model = model_instance
        self.device_used = device_used
        self.status_bar.setText(
            f"Status: Model loaded successfully on {self.device_used}. Ready.")
        self.generate_button.setEnabled(True)
        self.load_model_button.setEnabled(True)

    def on_model_load_error(self, error_msg):
        self.model = None
        self.status_bar.setText(f"Status: Model load failed. {error_msg}")
        self.generate_button.setEnabled(False)
        self.load_model_button.setEnabled(True)
        QMessageBox.critical(self, "Model Load Error", error_msg)

    def on_generation_thread_finished(self):
        print("UI: audio_generator_thread.finished signal received.")

        # Stop the timer regardless of how the thread finished
        if self.generation_timer.isActive():
            print("UI: Stopping generation timer.")
            self.generation_timer.stop()

        # Reset UI elements
        self.is_generating = False
        self.generate_button.setText("Generate Audio")
        self.generate_button.setEnabled(True)
        self.load_model_button.setEnabled(True)

        # Final status update based on how the thread might have ended,
        # if not already set by on_generation_complete or on_generation_error.
        # This ensures "Stopping..." doesn't linger.
        current_status = self.status_bar.text()
        if "stopping generation..." in current_status.lower() or \
           "stop requested." in current_status.lower():
            self.status_bar.setText("Status: Generation stopped by user.")
        elif not ("full audio generated" in current_status.lower() or
                  "failed" in current_status.lower() or
                  "stopped by user" in current_status.lower()):
            # If no specific completion or error message was set, default to Ready
            self.status_bar.setText("Status: Ready.")

    def update_generation_time_display(self):
        if self.generation_start_time and self.is_generating:
            elapsed_ms = self.generation_start_time.msecsTo(
                QTime.currentTime())
            # Only update if not showing chunk progress, to avoid flicker
            # and if the button still says "Stop Generation" (i.e. not "Stopping...")
            if "chunk" not in self.status_bar.text().lower() and \
               self.generate_button.text() == "Stop Generation":
                self.status_bar.setText(
                    f"Status: Generating... (Elapsed: {self.format_time(elapsed_ms)})")
        elif not self.is_generating and self.generation_timer.isActive():
            # This is a failsafe, should be stopped by on_generation_thread_finished
            print(
                "UI: Generation timer stopped by failsafe in update_generation_time_display.")
            self.generation_timer.stop()

    def update_generation_time(self):
        if self.generation_start_time and self.is_generating:
            elapsed_ms = self.generation_start_time.msecsTo(
                QTime.currentTime())
            seconds = int((elapsed_ms / 1000) % 60)
            minutes = int((elapsed_ms / (1000 * 60)) % 60)
            self.status_bar.setText(
                f"Status: Generating audio... {minutes:02}:{seconds:02}")

    def on_chunk_generated_progress(self, current_chunk, total_chunks):
        # This will be the primary status updater during active generation
        if not self.is_generating:
            return  # Don't update if we're trying to stop

        elapsed_str = ""
        if self.generation_start_time:
            elapsed_ms = self.generation_start_time.msecsTo(
                QTime.currentTime())
            elapsed_str = f" (Elapsed: {self.format_time(elapsed_ms)})"
        self.status_bar.setText(
            f"Status: Generating chunk {current_chunk}/{total_chunks}{elapsed_str}...")

    def on_generation_complete(self, output_path, sample_rate):
        # self.is_generating will be set to False by on_generation_thread_finished
        # self.generation_timer will be stopped by on_generation_thread_finished

        total_generation_time_str = ""
        if self.generation_start_time:
            elapsed_ms = self.generation_start_time.msecsTo(
                QTime.currentTime())
            total_generation_time_str = f" (Total time: {self.format_time(elapsed_ms)})"

        self.status_bar.setText(
            f"Status: Full audio generated: {os.path.basename(output_path)}{total_generation_time_str}")

        self.current_audio_file = output_path
        # ... (rest of the method same as your working version)
        self.current_file_label.setText(
            f"Last generated: {os.path.basename(output_path)}")
        
        self.media_player.setSource(QUrl.fromLocalFile(output_path))
        self.play_pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.playhead_slider.setEnabled(True)
        self.update_output_log()
        if self.autoplay_checkbox.isChecked():
            self.media_player.play()

    def on_generation_error(self, error_msg):
        # self.is_generating will be set to False by on_generation_thread_finished
        # self.generation_timer will be stopped by on_generation_thread_finished

        is_user_stop = "stopped by user" in error_msg.lower()

        final_status_msg = f"Status: {'Generation stopped by user.' if is_user_stop else 'Generation failed.'}"
        if not is_user_stop and error_msg:
            first_line_error = error_msg.splitlines()[0]
            if len(first_line_error) > 70:
                # Adjusted length
                first_line_error = first_line_error[:67] + "..."
            final_status_msg += f" ({first_line_error})"

        self.status_bar.setText(final_status_msg)

        if not is_user_stop:
            QMessageBox.critical(self, "Generation Error", error_msg)
        else:
            print(f"User stop confirmed by error signal: {error_msg}")

    def update_output_log(self):
        self.output_log_listwidget.clear()
        try:
            files = sorted([os.path.join(self.output_directory, f)for f in os.listdir(
                self.output_directory)if f.endswith(".wav")], key=os.path.getmtime, reverse=True)
            for f_path in files:
                item = QListWidgetItem(os.path.basename(f_path))
                item.setData(Qt.UserRole, f_path)
                self.output_log_listwidget.addItem(item)
        except Exception as e:
            print(f"Error updating output log: {e}")

    def play_selected_from_log(self, item: QListWidgetItem):
        file_path = item.data(Qt.UserRole)
        if file_path and os.path.exists(file_path):
            self.current_audio_file = file_path
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.current_file_label.setText(
                f"Playing from log: {os.path.basename(file_path)}")
            self.play_pause_button.setText("Pause")
            self.media_player.play()
            self.play_pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.playhead_slider.setEnabled(True)
        else:
            QMessageBox.warning(self, "File Error",
                                "Could not find or play selected audio file.")
            self.update_output_log()

    def toggle_play_pause(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            # Store current position before pausing
            self.paused_position = self.media_player.position()
            self.media_player.pause()
            # self.play_pause_button.setText("Play") # Done by handle_playback_state_changed
        else:  # Was Paused or Stopped
            if not self.current_audio_file or not os.path.exists(self.current_audio_file):
                QMessageBox.warning(
                    self, "No Audio", "No audio file loaded to play.")
                return

            # Ensure media source is set
            if self.media_player.source().isEmpty() or \
               self.media_player.source().toLocalFile() != self.current_audio_file:
                self.media_player.setSource(
                    QUrl.fromLocalFile(self.current_audio_file))
                # If source changed or was empty, assume play from start or last paused pos
                if hasattr(self, 'paused_position') and self.media_player.mediaStatus() != QMediaPlayer.MediaStatus.EndOfMedia:
                    self.media_player.setPosition(self.paused_position)
                else:  # Play from start if no paused_position or at end of media
                    self.media_player.setPosition(0)
                    self.paused_position = 0  # Reset paused position

            # If resuming from a paused state (and not end of media)
            elif hasattr(self, 'paused_position') and self.media_player.mediaStatus() != QMediaPlayer.MediaStatus.EndOfMedia:
                # Only set position if it's significantly different (avoids tiny jumps if already paused)
                # or if it was explicitly stopped and then play is hit again.
                if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
                    self.media_player.setPosition(self.paused_position)

            # If it was fully stopped (and not at end of media), or if no paused_position, play from current slider or 0
            elif self.media_player.playbackState() == QMediaPlayer.PlaybackState.StoppedState and \
                    self.media_player.mediaStatus() != QMediaPlayer.MediaStatus.EndOfMedia:
                # If stopped, it might have a valid position already (e.g. user seeked then stopped)
                # If no paused_position exists, it implies fresh play or play after stop
                if not hasattr(self, 'paused_position'):
                    self.media_player.setPosition(0)  # Default to start
                # else: play from current position (which might be 0 if stopped at start)

            self.media_player.play()
            # self.play_pause_button.setText("Pause") # Done by handle_playback_state_changed

    def stop_audio(self):
        self.media_player.stop()
        self.play_pause_button.setText("Play")

    # --- Slider Seeking Logic ---
    def slider_pressed(self): self.is_seeking_audio = True

    def seek_audio_on_move(self, position):
        if self.is_seeking_audio:
            self.media_player.setPosition(position)
            self.current_time_label.setText(self.format_time(position))

    def slider_released(self):
        if self.is_seeking_audio:
            self.is_seeking_audio = False
            self.media_player.setPosition(self.playhead_slider.value())

    def update_slider_position(self, position):
        if not self.is_seeking_audio:
            self.playhead_slider.setValue(position)
        self.current_time_label.setText(self.format_time(position))

    def update_duration_info(self, duration):
        if duration > 0:  # Only set range if duration is valid
            self.playhead_slider.setRange(0, duration)
            self.duration_label.setText(self.format_time(duration))
        else:  # Reset if duration is 0 or invalid (e.g. after stop or error)
            self.playhead_slider.setRange(0, 0)
            self.duration_label.setText("00:00")
            self.current_time_label.setText("00:00")

    def format_time(self, ms: int) -> str:  # Your improved version
        if ms < 0:
            ms = 0
        total_seconds_val = ms // 1000
        minutes_val = total_seconds_val // 60
        seconds_remainder_val = total_seconds_val % 60
        return f"{minutes_val:02}:{seconds_remainder_val:02}"

    def handle_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_pause_button.setText("Pause")
        else:
            self.play_pause_button.setText("Play")
        # This logic might conflict with explicit stop, if positionChanged handle slider reset mainly
        # if state==QMediaPlayer.PlaybackState.StoppedState and self.media_player.position()==0:   # Reset slider and time if stopped at start
        #     if self.media_player.mediaStatus()==QMediaPlayer.MediaStatus.EndOfMedia: # End of media reached
        #         self.playhead_slider.setValue(0);self.current_time_label.setText("00:00") # Reset to start

    def play_test_sound(self):
        """Generate and play a test tone to verify audio output"""
        print("Playing test sound...")
        import numpy as np
        
        # Generate a 1-second 440Hz sine wave
        sample_rate = 48000
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = np.sin(2 * np.pi * frequency * t)
        
        # Convert to torch tensor and save
        test_tensor = torch.from_numpy(test_audio).float().unsqueeze(0)
        test_path = os.path.join(self.output_directory, "test_tone.wav")
        
        # Save as 16-bit PCM
        test_tensor_int16 = (test_tensor * 32767).to(torch.int16)
        torchaudio.save(test_path, test_tensor_int16, sample_rate)
        
        print(f"Test tone saved to: {test_path}")
        self.media_player.setSource(QUrl.fromLocalFile(test_path))
        self.media_player.play()
        print(f"Audio output device: {self.audio_output.device().description()}")
        print(f"Volume: {self.audio_output.volume()}")
        print(f"Muted: {self.audio_output.isMuted()}")
    
    def update_volume(self, value):
        volume = value / 100.0
        self.audio_output.setVolume(volume)
        self.volume_label.setText(f"{value}%")
        print(f"Volume set to: {volume}")
    
    def handle_media_status_changed(self, status):
        print(f"Media status changed to: {status}")
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            print("Invalid media - audio format may not be supported")
        elif status == QMediaPlayer.MediaStatus.LoadedMedia:
            print("Media loaded successfully")
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            print("End of media reached")
    
    def handle_media_error(self):
        error_string = self.media_player.errorString()
        error_code = self.media_player.error()
        print(f"Media error occurred - Code: {error_code}, Message: {error_string}")
        if error_string:
            QMessageBox.warning(self, "Media Player Error",
                                f"Error playing audio: {error_string}")
        self.status_bar.setText("Status: Media player error.")
        self.play_pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.playhead_slider.setEnabled(False)
        self.update_duration_info(0)  # Reset duration display on error
    
    def _init_audio_output(self):
        """Initialize audio output with platform-specific handling"""
        # Get default device
        default_device = QMediaDevices.defaultAudioOutput()
        print(f"Default audio device: {default_device.description()}")
        
        # Create audio output
        self.audio_output = QAudioOutput(default_device)
        self.audio_output.setVolume(1.0)  # Max volume
        
        # Platform-specific audio configuration
        if sys.platform == "darwin":  # macOS
            print("macOS detected - configuring audio output for compatibility")
            # Ensure not muted (macOS sometimes has issues with mute state)
            if self.audio_output.isMuted():
                self.audio_output.setMuted(False)
                print("Unmuted audio output")
        
        print(f"Audio output muted: {self.audio_output.isMuted()}")

    def closeEvent(self, event):
        if hasattr(self, 'model_loader_thread') and self.model_loader_thread.isRunning():
            self.model_loader_thread.quit()
            self.model_loader_thread.wait()
        if hasattr(self, 'audio_generator_thread') and self.audio_generator_thread.isRunning():
            self.audio_generator_thread.stop()  # Request stop
            self.audio_generator_thread.wait()  # Wait for it to finish
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatterboxApp()
    window.show()
    sys.exit(app.exec())
