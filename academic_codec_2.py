import os
import argparse
import torch
import torchaudio
from academicodec.models.hificodec.vqvae_tester import VqvaeTester

def process_audio_files(input_dir, output_dir):
    # Set paths
    config_path = "/data/lekha_codec_model_files/AcademiCodec/egs/HiFi-Codec-24k-240d/config_24k_240d.json"
    model_path = "/data/lekha_codec_model_files/AcademiCodec/egs/HiFi-Codec-24k-240d/model.pt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the VqvaeTester
    vqvae_tester = VqvaeTester(config_path, model_path, sample_rate=24000)
    vqvae_tester.to(device)
    
    # Get all .wav files in input directory
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    
    for wav_file in wav_files:
        input_file_path = os.path.join(input_dir, wav_file)
        output_file_name = f"{os.path.splitext(wav_file)[0]}_academic_codec.wav"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        print(f"Processing {input_file_path}...")
        
        # Extract features/latents
        fid, vq_codes = vqvae_tester.vq(input_file_path)
        print(f"Extracted features for file: {fid}")
        print(f"Shape of extracted features: {vq_codes.shape}")
        
        # Regenerate audio from features
        _, reconstructed_audio = vqvae_tester.forward(input_file_path)
        
        # Convert to CPU and remove batch dimension
        reconstructed_audio = reconstructed_audio.cpu().squeeze(0)
        
        # Save the reconstructed audio
        torchaudio.save(output_file_path, reconstructed_audio, 24000)
        print(f"Reconstructed audio saved to: {output_file_path}")
    
    print("Processing complete.")
    

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process .wav files using AcademiCodec VQ-VAE model.")
    # parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing .wav files.")
    # parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory to save processed files.")
    # args = parser.parse_args()

    for personality in ["Joe_Biden","Kamala_Harris","Mathew_Miller","Vivek_Ramaswamy"]:
        input_dir = f"/data/FF_V2/Famous_Figures_V2/Data/No_Laundering/no_laundering_s16/{personality}/train/Original/"
        output_dir = f"/data/FF_V2/codec_fake_data/Famous_Figures_V2/Data/No_Laundering/no_laundering_s16/{personality}/train/academic_codec"
    
        process_audio_files(input_dir, output_dir)
