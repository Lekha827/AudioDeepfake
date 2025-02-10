import os
import argparse
import torch
import numpy as np
import soundfile as sf
from AudioDec.utils.audiodec import AudioDec

def process_audio_files(input_dir, output_dir):
    encoder_config_path = "audiodec_autoencoder_24k_320d/checkpoint-500000steps.pkl"
    decoder_config_path = "audiodec_vocoder_24k_320d/checkpoint-500000steps.pkl"
    
    tx_device = "cuda:2" if torch.cuda.is_available() else "cpu"
    rx_device = "cuda:2" if torch.cuda.is_available() else "cpu"
    
    audiodec = AudioDec(tx_device=tx_device, rx_device=rx_device)
    audiodec.load_transmitter(encoder_config_path)
    audiodec.load_receiver(encoder_config_path, decoder_config_path)
    
    sample_rate = 16000
    os.makedirs(output_dir, exist_ok=True)
    
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    
    for wav_file in wav_files:
        input_file_path = os.path.join(input_dir, wav_file)
        output_file_name = f"{os.path.splitext(wav_file)[0]}_audiodec_codec.wav"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        print(f"Processing {input_file_path}...")
        
        with torch.no_grad():
            if os.path.exists(input_file_path):
                data, fs = sf.read(input_file_path, always_2d=True)
            else:
                raise ValueError(f'Input file {input_file_path} does not exist!')
            
            assert fs == sample_rate, f"Data ({fs}Hz) is not matched to model ({sample_rate}Hz)!"
            x = np.expand_dims(data.transpose(1, 0), axis=1)  # (T, C) -> (C, 1, T)
            x = torch.tensor(x, dtype=torch.float).to(tx_device)
            
            print("Encode/Decode...")
            z = audiodec.tx_encoder.encode(x)
            idx = audiodec.tx_encoder.quantize(z)
            zq = audiodec.rx_encoder.lookup(idx)
            y = audiodec.decoder.decode(zq)[:, :, :x.size(-1)]
            y = y.squeeze(1).transpose(1, 0).cpu().numpy()  # T x C
            
            sf.write(output_file_path, y, fs, "PCM_16")
            print(f"Saved processed file to {output_file_path}")
    
    print("Processing complete.")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process .wav files using AudioDec codec.")
    # parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing .wav files.")
    # parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory to save processed files.")
    # args = parser.parse_args()

    for personality in ["Kamala_Harris","Mathew_Miller","Vivek_Ramaswamy"]:
        input_dir = f"/data/FF_V2/Famous_Figures_V2/Data/No_Laundering/no_laundering_s16/{personality}/train/Original/"
        output_dir = f"/data/FF_V2/codec_fake_data/Famous_Figures_V2/Data/No_Laundering/no_laundering_s16/{personality}/train/audio_dec_codec"
    
        process_audio_files(input_dir, output_dir)
