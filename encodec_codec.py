import os
import argparse
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

def process_audio_files(input_dir, output_dir):
    # Instantiate a pretrained EnCodec model
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)

    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .wav files in input directory
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    
    for wav_file in wav_files:
        input_file_path = os.path.join(input_dir, wav_file)
        output_file_name = f"{os.path.splitext(wav_file)[0]}_encodec.wav"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        print(f"Processing {input_file_path}...")
        
        # Load and pre-process the audio waveform
        wav, sr = torchaudio.load(input_file_path)
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.unsqueeze(0)
        
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        
        # Generate audio from codes
        with torch.no_grad():
            audio_values = model.decode([(codes, None)])
        
        # Convert the decoded audio tensor to numpy array
        audio_values = audio_values.squeeze(0)
        
        # Save the decoded audio
        torchaudio.save(output_file_path, audio_values.cpu(), model.sample_rate)
        print(f"Saved processed file to {output_file_path}")
    
    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .wav files using EnCodec model.")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory containing .wav files.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory to save processed files.")
    args = parser.parse_args()
    
    process_audio_files(args.input_dir, args.output_dir)
