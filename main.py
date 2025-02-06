import dac
from audiotools import AudioSignal
import os

# Download a model
model_path = dac.utils.download(model_type="44khz")

model = dac.DAC.load(model_path).to("cuda")
print(model.device)
current_dir = os.getcwd()

# Load audio signal file
file_path = os.path.join(current_dir, "sample_9.wav")
female_voice_file_path = os.path.join(current_dir, "female_voice.wav")
paths = [file_path, female_voice_file_path]
signal = AudioSignal(female_voice_file_path)

# Encode audio signal as one long file
# (may run out of GPU memory on long files)
signal.to(model.device)

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

#Write to file
output_file_name = "dac_output_female_voice.wav"
y.write(output_file_name)
# for path in paths:
#     signal = AudioSignal(path)

#     # Encode audio signal as one long file
#     # (may run out of GPU memory on long files)
#     signal.to(model.device)

#     x = model.preprocess(signal.audio_data, signal.sample_rate)
#     z, codes, latents, _, _ = model.encode(x)

#     # Decode audio signal
#     y = model.decode(z)

#     # Alternatively, use the `compress` and `decompress` functions
#     # to compress long files.

#     signal = signal.cpu()
#     x = model.compress(signal)

#     # Save and load to and from disk
#     x.save("compressed.dac")
#     x = dac.DACFile.load("compressed.dac")

#     # Decompress it back to an AudioSignal
#     y = model.decompress(x)

#     #Write to file
#     output_file_name = "dac_output_"+ path.split('/')[-1]
#     y.write(output_file_name)