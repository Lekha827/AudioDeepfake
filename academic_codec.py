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
    output_filename = os.path.join(output_dir, f"{filename}_academic_codec.wav")

    import pdb; pdb.set_trace()
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)  # This line ensures the output directory exists
    
    # Extract features/latents
    fid, vq_codes = vqvae_tester.vq(audio_path)

    # Regenerate audio from features
    _, reconstructed_audio = vqvae_tester.forward(audio_path)

    # Convert to CPU and remove batch dimension
    reconstructed_audio = reconstructed_audio.cpu().squeeze(0)

    # Save the reconstructed audio
    torchaudio.save(output_filename, reconstructed_audio, 16000)
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
    vqvae_tester = VqvaeTester(config_path, model_path, sample_rate=16000)
    vqvae_tester.cuda()  # Move to GPU if available
    
    # Get all .wav files in input directory
    wav_files = [f for f in os.listdir(args.input_dir) if f.endswith(".wav")]

    if not wav_files:
        print(f"No .wav files found in {args.input_dir}. Exiting...")
        return
    
    
    for wav_file in wav_files:
        input_file_path = os.path.join(args.input_dir, wav_file)
        
        print(f"Processing {input_file_path}...")
        process_audio(input_file_path, vqvae_tester, args.output_dir)

if __name__ == "__main__":
    main()
