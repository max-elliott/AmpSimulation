import librosa
import numpy as np
import os

def resample_directory(input_dir, output_dir, old_sr, new_sr):
    for file in librosa.util.find_files(input_dir):
        print(file)
        wav, sr = librosa.load(os.path.join(input_dir, file), sr=old_sr)
        print(wav.shape, sr)
        resampled_wav = librosa.core.resample(wav, sr, new_sr)

        output_name = os.path.basename(file).replace('.wav', f'_{new_sr}.wav')
        output_path = os.path.join(output_dir, output_name)
        librosa.output.write_wav(output_path, resampled_wav, new_sr)
        print(f"Saved as {output_name}")

if __name__ == '__main__':

    input_dir = '/Users/Max/Documents/AI/AmpSimulation/data/raw'
    output_dir = '/Users/Max/Documents/AI/AmpSimulation/data/resampled'
    new_sr = 22050
    # resample_directory(input_dir, output_dir, 44100, new_sr)

    audio_dir = '/Users/Max/Documents/AI/AmpSimulation/data/re'
    resampled_files = librosa.util.find_files(audio_dir)
    print(resampled_files)
    output_dir = '/Users/Max/Documents/AI/AmpSimulation/data/re/audio'
    min_length = np.min([librosa.load(wav, sr=new_sr)[0].shape[0] for wav in resampled_files])
    print(min_length)
    segment_length_seconds = 4
    segment_length_samples = segment_length_seconds * new_sr
    num_segments = min_length // (segment_length_seconds * new_sr)
    num_segments = 10
    for file in resampled_files:
        wav, sr = librosa.load(file, sr=new_sr)
        print(wav.shape)
        filename_index = 0
        for i in range(segment_length_samples + 1, min_length, segment_length_samples):
            cropped_wav = wav[i - segment_length_samples:i]
            output_name = os.path.basename(file).replace('.wav', f'_{filename_index:05d}.wav')
            output_path = os.path.join(output_dir, output_name)
            librosa.output.write_wav(output_path, cropped_wav, new_sr)
            filename_index += 1
