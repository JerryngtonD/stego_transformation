from pydub import AudioSegment


def convert_to_mp3(file_path):
    AudioSegment.from_mp3(file_path).export("/output/file.wav", format="wav")
    print("Converting to wav was finished")
