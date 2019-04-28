from pydub import AudioSegment


def convert_to_mp3(file_path):
    AudioSegment.from_wav(file_path).export("/output/file.mp3", format="mp3")
    print("Converting to mp3 was finished")
