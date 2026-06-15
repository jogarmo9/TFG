"""
silero_vad_validator.py
=======================
Silero VAD validator module for filtering false positive Speech predictions.

Uses Silero VAD to validate YOLO Speech detections.
If confidence is low AND VAD disagrees → likely false positive → filter out.

Silero VAD: https://github.com/snakers4/silero-vad
Install: pip install silero-vad
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

# Silero VAD operates at 16kHz
VAD_SR = 16000

# Silero VAD confidence threshold
VAD_THRESHOLD = 0.5  # probability threshold (0-1)


class SileroVADValidator:
    """
    Validates YOLO Speech predictions using Silero VAD.
    
    Logic:
    - If YOLO_confidence >= 0.5: ACCEPT (strong prediction)
    - If YOLO_confidence < 0.3: Run VAD on audio segment
      - If VAD_prob >= 0.5: ACCEPT (VAD confirms speech)
      - If VAD_prob < 0.5: REJECT (likely noise)
    - If 0.3 <= YOLO_confidence < 0.5: ACCEPT with warning (borderline)
    """
    
    def __init__(self):
        """Initialize Silero VAD model from torch.hub."""
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True
            )
            self.get_speech_timestamps, _, self.read_audio, _, self.collect_chunks = self.utils
            print("✓ Silero VAD loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD: {e}")
    
    def validate_segment(self, audio: np.ndarray, sr: int = VAD_SR,
                        yolo_confidence: float = 0.1,
                        debug: bool = False) -> tuple[bool, float]:
        """
        Validate a speech segment using Silero VAD.
        
        Args:
            audio: numpy array of audio samples (mono, 16-bit)
            sr: sample rate (should be 16000)
            yolo_confidence: YOLO confidence for this prediction
            debug: print debug info
        
        Returns:
            (should_keep: bool, vad_prob: float)
        """
        if sr != VAD_SR:
            raise ValueError(f"Silero VAD requires 16kHz audio, got {sr}Hz")
        
        # Normalize audio to [-1, 1]
        audio_float = audio.astype(np.float32)
        if np.max(np.abs(audio_float)) > 1.0:
            audio_float = audio_float / 32768.0  # 16-bit PCM
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_float)
        
        try:
            # Get VAD timestamps
            timestamps = self.get_speech_timestamps(audio_tensor, self.model, sampling_rate=VAD_SR)
            
            # Calculate VAD probability as fraction of audio with speech
            if len(timestamps) == 0:
                vad_prob = 0.0
            else:
                total_speech_samples = sum(ts['end'] - ts['start'] for ts in timestamps)
                vad_prob = min(1.0, total_speech_samples / len(audio_tensor))
        
        except Exception as e:
            if debug:
                print(f"  [VAD ERROR] {e}")
            vad_prob = 0.0
        
        # Decision logic
        if yolo_confidence >= 0.5:
            # Strong YOLO prediction → ACCEPT
            decision = True
            reason = "YOLO_strong"
        elif yolo_confidence < 0.3:
            # Weak YOLO prediction → trust VAD
            decision = vad_prob >= VAD_THRESHOLD
            reason = "VAD_validation"
        else:  # 0.3 <= confidence < 0.5
            # Borderline → ACCEPT with warning
            decision = True
            reason = "borderline"
        
        if debug:
            print(f"    YOLO_conf={yolo_confidence:.3f}, VAD_prob={vad_prob:.3f} "
                  f"→ {['REJECT', 'ACCEPT'][int(decision)]} ({reason})")
        
        return decision, vad_prob


# ──────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Silero VAD validator...")
    
    try:
        vad = SileroVADValidator()
        print("✓ Silero VAD initialized successfully")
        
        # Test on dummy audio
        test_audio = np.random.randn(VAD_SR * 2).astype(np.float32) * 0.05  # 2 sec of noise
        decision, prob = vad.validate_segment(test_audio, debug=True)
        print(f"✓ Test prediction (noise): {decision}, VAD_prob={prob:.3f}")
        
        # Test on louder audio (should look more like speech)
        test_audio2 = (np.sin(2*np.pi*440*np.linspace(0, 2, VAD_SR*2)) * 0.3).astype(np.float32)
        decision2, prob2 = vad.validate_segment(test_audio2, yolo_confidence=0.2, debug=True)
        print(f"✓ Test prediction (tone): {decision2}, VAD_prob={prob2:.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
