import dac
from audiotools import AudioSignal
import torch
import os
import argparse

def process_audio_files(input_dir, output_dir):
    # Download the model
    model_path = dac.utils.download(model_type="16khz")
    model = dac.DAC.load(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .wav files in input directory
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    
    for wav_file in wav_files:
        input_file_path = os.path.join(input_dir, wav_file)
        output_file_name = f"{os.path.splitext(wav_file)[0]}_dac_codec.wav"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        print(f"Processing {input_file_path}...")
        
        # Load audio signal
        signal = AudioSignal(input_file_path)
        signal.to(model.device)
        
        # Encode audio signal
        x = model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, latents, _, _ = model.encode(x)
        
        # Decode audio signal
        y = model.decode(z)

        # Alternatively, use the `compress` and `decompress` functions
        # to compress long files.

        signal = signal.cpu()
        x = model.compress(signal)

        # Save and load to and from disk
        x.save("compressed.dac")
        x = dac.DACFile.load("compressed.dac")

        # Decompress it back to an AudioSignal
        y = model.decompress(x)

        
        # Save output file
        y.write(output_file_path)
        print(f"Saved processed file to {output_file_path}")
    
    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .wav files using DAC codec.")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory containing .wav files.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory to save processed files.")
    args = parser.parse_args()
    
    process_audio_files(args.input_dir, args.output_dir)
