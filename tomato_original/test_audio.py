import os
import sys
import numpy as np
import librosa

from src.audio.recognizer import AudioRecognizer

def test_model_with_file(file_path):
    print(f"Testing with file: {file_path}")
    recognizer = AudioRecognizer()
    
    if recognizer.interpreter is None:
        print("Model not loaded!")
        return

    # Load audio file directly using librosa
    sample_rate = 22050
    audio_data, _ = librosa.load(file_path, sr=sample_rate, duration=3.0)
    
    # Extract features using recognizer's method
    features = recognizer.extract_mel_spectrogram(audio_data, sample_rate)
    
    if features is None:
        print("Feature extraction failed.")
        return
        
    recognizer.interpreter.set_tensor(recognizer.input_details[0]['index'], features)
    recognizer.interpreter.invoke()
    output_data = recognizer.interpreter.get_tensor(recognizer.output_details[0]['index'])[0]
    
    if recognizer.output_details[0]['dtype'] == np.int8:
        scale, zero_point = recognizer.output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - zero_point) * scale
        
    probs = output_data
    max_prob_index = np.argmax(probs)
    confidence = probs[max_prob_index]
    
    predicted_label = recognizer.labels[max_prob_index] if max_prob_index < len(recognizer.labels) else "Unknown"
    
    print("Probabilities:")
    for i, label in enumerate(recognizer.labels):
        print(f"  {label}: {probs[i]:.4f}")
        
    print(f"\nResult: Predicted '{predicted_label}' with {confidence:.4f} confidence.")

if __name__ == "__main__":
    test_file = os.path.join('dataset', 'audio', 'Sparrow', 'Sparrow001.mp3')
    test_model_with_file(test_file)
