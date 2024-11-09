class BalladResults:
  def classify_genre(filepath):
    def load_genre_classifier():
      from datasets import load_dataset, Audio
      import numpy as np
      from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
      import evaluate
      option = input("Do you want to import the genre classifier from your Google Drive? (y/n): ")
      if option == 'y':
        from google.colab import drive
        drive.mount('/content/drive')
      path = input("Enter path to the saved model folder (Eg. /content/drive/MyDrive/Saved_Model): ")
      loaded_model = AutoModelForAudioClassification.from_pretrained(path)
      loaded_feature_extractor = AutoFeatureExtractor.from_pretrained(path)
      return loaded_model, loaded_feature_extractor

    loaded_model, loaded_feature_extractor = load_genre_classifier()
    from transformers import pipeline, AutoFeatureExtractor
    import pretty_midi

    pipe = pipeline("audio-classification", model=loaded_model,
                    feature_extractor=loaded_feature_extractor, device=-1)  # Use GPU if available

    def classify_audio(filepath):
        try:
            # Convert MIDI to WAV if needed
            if filepath.endswith('.mid'):
                midi_data = pretty_midi.PrettyMIDI(filepath)
                audio_data = midi_data.fluidsynth()  # Synthesize audio from MIDI
                # Save as temporary WAV file
                temp_wav_file = "temp.wav"
                import soundfile as sf
                sf.write(temp_wav_file, audio_data, samplerate=44100)
                filepath = temp_wav_file

            preds = pipe(filepath)
            outputs = {}
            for p in preds:
                outputs[p["label"]] = p["score"]
            return outputs
        except ValueError as e:
            print(f"Error processing audio file: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    output = classify_audio(filepath)

    if output:
        print("Predicted Genre:")
        max_key = max(output, key=output.get)
        print("The predicted genre is:", max_key)
        print("The prediction score is:", output[max_key])
        print("\n\n")
    else:
      print("No audio file found.")

  def CLAP_similarity(q, audio_path):
    from transformers import ClapModel, ClapProcessor
    import torch
    import torchaudio
    import numpy as np
    import random

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    set_seed(42)

    model = ClapModel.from_pretrained("laion/clap-htsat-fused")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

    def process_audio(audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 48000:
            resampler = torchaudio.transforms.Resample(sample_rate, 48000)
            waveform = resampler(waveform)
        return waveform.squeeze().numpy()

    def get_embeddings(texts, audio_path, processor, model):
        # Process audio
        audio = process_audio(audio_path)

        # Prepare inputs
        inputs = processor(
            text=texts,
            audios=audio,
            return_tensors="pt",
            padding=True,

        )
        inputs["is_longer"] = torch.zeros(len(texts),dtype = torch.bool)
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Normalize embeddings
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        audio_embeds = outputs.audio_embeds / outputs.audio_embeds.norm(dim=-1, keepdim=True)

        return text_embeds, audio_embeds

    def calculate_similarity(text_embeds, audio_embeds):
        similarities = torch.matmul(text_embeds, audio_embeds.T).squeeze(1)
        return similarities

    def run_test(prompts, audio_path):
        if audio_path.endswith('.mid'):
          midi_data = pretty_midi.PrettyMIDI(audio_path)
          audio_data = midi_data.fluidsynth()  # Synthesize audio from MIDI
          # Save as temporary WAV file
          temp_wav_file = f"{audio_path[:-4]}.wav"
          import soundfile as sf
          sf.write(temp_wav_file, audio_data, samplerate=44100)
          audio_path = temp_wav_file

          text_embeds, audio_embeds = get_embeddings(prompts, audio_path, processor, model)
          similarities = calculate_similarity(text_embeds, audio_embeds)

          for prompt, similarity in zip(prompts, similarities):
              if prompt == q:
                print(f"Your Prompt: '{prompt}'\nSimilarity: {similarity.item():.4f}\n")
              else:
                print(f"Prompt: '{prompt}'\nSimilarity: {similarity.item():.4f}\n")

    prompts = [
        q,
        "jolly piano music",
        "Heavy metal rock music",
        "Acoustic guitar piece",
        "Birds chirping in a forest",
        "A busy street with cars honking",
        "A person speaking about politics"
    ]

    run_test(prompts, audio_path)
    
    
    
BalladResults.classify_genre("temp0.mid")
BalladResults.CLAP_similarity(q, "temp0.mid")