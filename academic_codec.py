import glob
import os
import argparse
import torch
import torchaudio
from academicodec.models.hificodec.vqvae_tester import VqvaeTester

def process_audio(audio_path, vqvae_tester, output_dir):
    """Process a single audio file and save the reconstructed output."""
    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = f"output_dir/{filename}_academic_codec.wav"
    
    # Extract features/latents
    fid, vq_codes = vqvae_tester.vq(audio_path)
    print(f"Extracted features for file: {audio_path}")
    print(f"Shape of extracted features: {vq_codes.shape}")
    
    # Regenerate audio from features
    _, reconstructed_audio = vqvae_tester.forward(audio_path)
    
    # Convert to CPU and remove batch dimension
    reconstructed_audio = reconstructed_audio.cpu().squeeze(0)
    
    # Save the reconstructed audio
    torchaudio.save(output_filename, reconstructed_audio, 24000)
    print(f"Reconstructed audio saved to: {output_filename}")

def main():
    """Parse command-line arguments and process multiple audio files."""
    parser = argparse.ArgumentParser(description="Process multiple audio files using AcademiCodec.")
    parser.add_argument("--input_dir", help="List of input audio file paths.", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    # Set paths for the model
    config_path = "/data/lekha_codec_model_files/AcademiCodec/egs/HiFi-Codec-24k-240d/config_24k_240d.json"
    model_path = "/data/lekha_codec_model_files/AcademiCodec/egs/HiFi-Codec-24k-240d/model.pt"
    
    # Initialize the VqvaeTester
    vqvae_tester = VqvaeTester(config_path, model_path, sample_rate=24000)
    vqvae_tester.cuda()  # Move to GPU if available

    # Find all .wav files in input_dir
    wav_files = glob.glob(os.path.join(args.input_dir, "*.wav"))

    if not wav_files:
        print(f"No .wav files found in {args.input_dir}. Exiting...")
        return
    
    # Process each audio file in the list
    for path in args.input_dir:
        process_audio(path, vqvae_tester, args.output_dir)

if __name__ == "__main__":
    main()
