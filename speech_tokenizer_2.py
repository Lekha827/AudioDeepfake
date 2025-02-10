import os
import argparse
import torch
import torchaudio
from speechtokenizer import SpeechTokenizer

def process_audio_files(input_dir, output_dir):
    # Load SpeechTokenizer model
    config_path = '/data/lekha_codec_model_files/config.json'
    ckpt_path = '/data/lekha_codec_model_files/SpeechTokenizer.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    model.to(device)
    model.eval()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .wav files in input directory
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    
    for wav_file in wav_files:
        input_file_path = os.path.join(input_dir, wav_file)
        output_file_name = f"{os.path.splitext(wav_file)[0]}_speech_tokenizer.wav"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        print(f"Processing {input_file_path}...")
        
        # Load and preprocess audio
        wav, sr = torchaudio.load(input_file_path)
        if wav.shape[0] > 1:
            wav = wav[:1, :]
        if sr != model.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
        wav = wav.unsqueeze(0).to(device)
        
        # Extract discrete codes
        with torch.no_grad():
            codes = model.encode(wav)  # codes: (n_q, B, T)
        
        RVQ_1 = codes[:1, :, :]
        RVQ_supplement = codes[1:, :, :]
        
        # Combine and reconstruct waveform
        reconstructed_codes = torch.cat([RVQ_1, RVQ_supplement], dim=0).to(device)
        with torch.no_grad():
            reconstructed_wav = model.decode(reconstructed_codes)
        
        # Ensure valid range and save audio
        reconstructed_wav = reconstructed_wav.squeeze(0).clamp(-1.0, 1.0).cpu()
        torchaudio.save(output_file_path, reconstructed_wav, model.sample_rate)
        print(f"Saved processed file to {output_file_path}")
    
    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .wav files using SpeechTokenizer model.")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory containing .wav files.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory to save processed files.")
    args = parser.parse_args()
    
    process_audio_files(args.input_dir, args.output_dir)
