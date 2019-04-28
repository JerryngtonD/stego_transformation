from pydub import AudioSegment


def convert_to_wav():
    sound = AudioSegment.from_file('/Users/evgeny/PycharmProjects/stego/output/file.mp3')
    sound.export("/Users/evgeny/PycharmProjects/stego/output/result_file.wav", format="wav")
    print("Converting to wav was finished")

convert_to_mp3()