import note_seq
from note_seq.protobuf import music_pb2
import matplotlib.pyplot as plt
from note_seq.midi_io import note_sequence_to_midi_file
import IPython.display as ipd
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForCausalLM
import mido
import subprocess
from langchain_google_genai import ChatGoogleGenerativeAI
import os
tokenizer = AutoTokenizer.from_pretrained("ai-guru/lakhclean_mmmtrack_4bars_d-2048")
model = AutoModelForCausalLM.from_pretrained("ai-guru/lakhclean_mmmtrack_4bars_d-2048")
os.environ["GOOGLE_API_KEY"] = ''

class Ballad:
  def empty_note_sequence(qpm=120.0, total_time=0.0):
    note_sequence = note_seq.protobuf.music_pb2.NoteSequence()
    note_sequence.tempos.add().qpm = qpm
    note_sequence.ticks_per_quarter = note_seq.constants.STANDARD_PPQ
    note_sequence.total_time = total_time
    return note_sequence
  
  def midi_wav(midi_file,wav_file):
    fs = FluidSynth()
    fs.midi_to_audio(midi_file, wav_file)
  
  def token_sequence_to_note_sequence(token_sequence, use_program=True, use_drums=True, instrument_mapper=None, only_piano=False):
    NOTE_LENGTH_16TH_120BPM = 0.25 * 60 / 120
    BAR_LENGTH_120BPM = 4.0 * 60 / 120
    if isinstance(token_sequence, str):
        token_sequence = token_sequence.split()

    note_sequence = Ballad.empty_note_sequence()

    # Render all notes.
    current_time = 0
    current_program = 1
    current_is_drum = False
    current_instrument = 0
    track_count = 0
    current_bar_index = 0
    for token_index, token in enumerate(token_sequence):
        current_notes={}
        if token == "PIECE_START":
            pass
        elif token == "PIECE_END":
            print("The end.")
            break
        elif token == "TRACK_START":
            current_bar_index = 0
            track_count += 1
            pass
        elif token == "TRACK_END":
            pass
        elif token == "KEYS_START":
            pass
        elif token == "KEYS_END":
            pass
        elif token.startswith("KEY="):
            pass
        elif token.startswith("INST"):
            instrument = token.split("=")[-1]
            if instrument != "DRUMS" and use_program:
                if instrument_mapper is not None:
                    if instrument in instrument_mapper:
                        instrument = instrument_mapper[instrument]
                current_program = int(instrument)
                current_instrument = track_count
                current_is_drum = False
            if instrument == "DRUMS" and use_drums:
                current_instrument = 0
                current_program = 0
                current_is_drum = True
        elif token == "BAR_START":
            current_time = current_bar_index * BAR_LENGTH_120BPM
            current_notes = {}
        elif token == "BAR_END":
            current_bar_index += 1
            pass
        elif token.startswith("NOTE_ON"):
            pitch = int(token.split("=")[-1])
            note = note_sequence.notes.add()
            note.start_time = current_time
            note.end_time = current_time + 4 * NOTE_LENGTH_16TH_120BPM
            note.pitch = pitch
            note.instrument = current_instrument
            note.program = current_program
            note.velocity = 80
            note.is_drum = current_is_drum
            current_notes[pitch] = note
        elif token.startswith("NOTE_OFF"):
            pitch = int(token.split("=")[-1])
            if pitch in current_notes:
                note = current_notes[pitch]
                note.end_time = current_time
        elif token.startswith("TIME_DELTA"):
            delta = float(token.split("=")[-1]) * NOTE_LENGTH_16TH_120BPM
            current_time += delta
        elif token.startswith("DENSITY="):
            pass
        elif token == "[PAD]":
            pass
        else:
            #print(f"Ignored token {token}.")
            pass

    # Make the instruments right.
    instruments_drums = []
    for note in note_sequence.notes:
        pair = [note.program, note.is_drum]
        if pair not in instruments_drums:
            instruments_drums += [pair]
        note.instrument = instruments_drums.index(pair)

    if only_piano:
        for note in note_sequence.notes:
            if not note.is_drum:
                note.instrument = 0
                note.program = 0

    return note_sequence
  
  def generate_fuckall1(inst, n,t, number=1):
    if number not in range(1, 5):
        print("Only up to 4 generations are valid")
        return None
    else:
        res=[]
        for song_num in range(number):
            generated_sequence = "PIECE_START PIECE_START TRACK_START"

            # Add all instruments to the initial sequence
            inst=Ballad.instrument_mapper()
            inst=list(map(int,inst.split()))
            for i in inst:
                generated_sequence += " INST=" + str(i)

            input_ids = tokenizer.encode(generated_sequence, return_tensors="pt")
            eos_token_id = tokenizer.encode("TRACK_END")[0]
            temperature = t

            # Generate the initial sequence
            generated_ids = model.generate(
                input_ids,
                max_length=16384,
                do_sample=True,
                temperature=temperature,
                eos_token_id=eos_token_id
            )
            generated_sequence = tokenizer.decode(generated_ids[0])

            # Perform additional generations if n > 1
            if n > 1:
                generated_sequence = generated_sequence[:-10]  # Remove the last TRACK_END
                for _ in range(n - 1):
                    input_ids = tokenizer.encode(generated_sequence, return_tensors="pt")
                    generated_ids = model.generate(
                        input_ids,
                        max_length=8192,
                        do_sample=True,
                        temperature=temperature,
                        eos_token_id=eos_token_id
                    )
                    generated_sequence = tokenizer.decode(generated_ids[0])

            print(f"Generated sequence for song {song_num + 1}:")
            print(generated_sequence)

            try:
                note_sequence = Ballad.token_sequence_to_note_sequence(generated_sequence)
                #synth = note_seq.fluidsynth
                note_seq.midi_io.note_sequence_to_midi_file(note_sequence, f'static/audio/temp{song_num}.mid')
                print(f"Generated and saved temp{song_num}.mid")
                Ballad.midi_wav(f"static/audio/temp{song_num}.mid",f"static/audio/temp{song_num}.wav")
                res.append(f"static/audio/temp{song_num}.wav")
                #note_seq.play_sequence(note_sequence, synth)
            except Exception as e:
                print(f"Error generating song {song_num + 1}, try again:", e)
    return res

  def instrument_mapper(q):
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    if q=='end':
      return 'stopped'
    else:
      prompt = f"""You can only reply to questions by providing the index of the instrument according to the General MIDI format. The dictionary for the same is provided
      "acoustic grand piano": 0, "bright acoustic piano": 1, "electric grand piano": 2, "honky-tonk piano": 3,
      "electric piano 1": 4, "electric piano 2": 5, "harpsichord": 6, "clavinet": 7, "celesta": 8, "glockenspiel": 9,
      "music box": 10, "vibraphone": 11, "marimba": 12, "xylophone": 13, "tubular bells": 14, "dulcimer": 15,
      "drawbar organ": 16, "percussive organ": 17, "rock organ": 18, "church organ": 19, "reed organ": 20,
      "accordion": 21, "harmonica": 22, "tango accordion": 23, "acoustic guitar (nylon)": 24, "acoustic guitar (steel)": 25,
      "electric guitar (jazz)": 26, "electric guitar (clean)": 27, "electric guitar (muted)": 28, "overdriven guitar": 29,
      "distortion guitar": 30, "guitar harmonics": 31, "acoustic bass": 32, "electric bass (finger)": 33,
      "electric bass (pick)": 34, "fretless bass": 35, "slap bass 1": 36, "slap bass 2": 37, "synth bass 1": 38,
      "synth bass 2": 39, "violin": 40, "viola": 41, "cello": 42, "contrabass": 43, "tremolo strings": 44,
      "pizzicato strings": 45, "orchestral harp": 46, "timpani": 47, "string ensemble 1": 48, "string ensemble 2": 49,
      "synthstrings 1": 50, "synthstrings 2": 51, "choir aahs": 52, "voice oohs": 53, "synth voice": 54, "orchestra hit": 55,
      "trumpet": 56, "trombone": 57, "tuba": 58, "muted trumpet": 59, "french horn": 60, "brass section": 61,
      "synthbrass 1": 62, "synthbrass 2": 63, "soprano sax": 64, "alto sax": 65, "tenor sax": 66, "baritone sax": 67,
      "oboe": 68, "english horn": 69, "bassoon": 70, "clarinet": 71, "piccolo": 72, "flute": 73, "recorder": 74,
      "pan flute": 75, "blown bottle": 76, "shakuhachi": 77, "whistle": 78, "ocarina": 79, "lead 1 (square)": 80,
      "lead 2 (sawtooth)": 81, "lead 3 (calliope)": 82, "lead 4 (chiff)": 83, "lead 5 (charang)": 84, "lead 6 (voice)": 85,
      "lead 7 (fifths)": 86, "lead 8 (bass + lead)": 87, "pad 1 (new age)": 88, "pad 2 (warm)": 89, "pad 3 (polysynth)": 90,
      "pad 4 (choir)": 91, "pad 5 (bowed)": 92, "pad 6 (metallic)": 93, "pad 7 (halo)": 94, "pad 8 (sweep)": 95,
      "fx 1 (rain)": 96, "fx 2 (soundtrack)": 97, "fx 3 (crystal)": 98, "fx 4 (atmosphere)": 99, "fx 5 (brightness)": 100,
      "fx 6 (goblins)": 101, "fx 7 (echoes)": 102, "fx 8 (sci-fi)": 103, "sitar": 104, "banjo": 105, "shamisen": 106,
      "koto": 107, "kalimba": 108, "bagpipe": 109, "fiddle": 110, "shanai": 111, "tinkle bell": 112, "agogo": 113,
      "steel drums": 114, "woodblock": 115, "taiko drum": 116, "melodic tom": 117, "synth drum": 118,
      "reverse cymbal": 119, "guitar fret noise": 120, "breath noise": 121, "seashore": 122, "bird tweet": 123,
      "telephone ring": 124, "helicopter": 125, "applause": 126, "gunshot": 127
        Say the input of the of the user doesn't have the exact keyword as that of the keys in the dictionary take the closest sounding instrument and return the index for it.
        The user query will contain names of instruments, and you need to return their corresponding indexes where each index is serperated by a space. If the user gives an emotion or vibe, generate as many instrument indexes as you think corresponds to the query.
        Generate a maximum of 44 instruments. If a user query is not correlated to any instrument, generate 4 random instruments. Do not generate more than 4 instruments for any use case.
        Here is my query: {q}
        """

      result = llm.invoke(prompt)
      return result

  def generate_fuckall(inst, n,t, number=1):
    if number not in range(1, 5):
        print("Only up to 4 generations are valid")
        return None
    else:
        res=[]
        for song_num in range(number):
            generated_sequence = "PIECE_START PIECE_START TRACK_START"

            # Add all instruments to the initial sequence
            for i in inst:
                generated_sequence += " INST=" + str(i)

            input_ids = tokenizer.encode(generated_sequence, return_tensors="pt")
            eos_token_id = tokenizer.encode("TRACK_END")[0]
            temperature = t

            # Generate the initial sequence
            generated_ids = model.generate(
                input_ids,
                max_length=16384,
                do_sample=True,
                temperature=temperature,
                eos_token_id=eos_token_id
            )
            generated_sequence = tokenizer.decode(generated_ids[0])

            # Perform additional generations if n > 1
            if n > 1:
                generated_sequence = generated_sequence[:-10]  # Remove the last TRACK_END
                for _ in range(n - 1):
                    input_ids = tokenizer.encode(generated_sequence, return_tensors="pt")
                    generated_ids = model.generate(
                        input_ids,
                        max_length=8192,
                        do_sample=True,
                        temperature=temperature,
                        eos_token_id=eos_token_id
                    )
                    generated_sequence = tokenizer.decode(generated_ids[0])

            print(f"Generated sequence for song {song_num + 1}:")
            print(generated_sequence)

            try:
                note_sequence = Ballad.token_sequence_to_note_sequence(generated_sequence)
                #synth = note_seq.fluidsynth
                note_seq.midi_io.note_sequence_to_midi_file(note_sequence, f'static/audio/temp{song_num}.mid')
                print(f"Generated and saved temp{song_num}.mid")
                Ballad.midi_wav(f"static/audio/temp{song_num}.mid",f"static/audio/temp{song_num}.wav")
                res.append(f"static/audio/temp{song_num}.wav")
                #note_seq.play_sequence(note_sequence, synth)
            except Exception as e:
                print(f"Error generating song {song_num + 1}, try again:", e)
    return res
