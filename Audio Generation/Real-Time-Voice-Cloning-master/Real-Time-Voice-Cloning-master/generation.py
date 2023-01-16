import sys
import traceback
from pathlib import Path
from time import perf_counter as timer

import numpy as np
import torch
import sounddevice as sd

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from toolbox.ui import UI
from toolbox.utterance import Utterance
from vocoder import inference as vocoder
import os
from warnings import warn
from pygame import mixer
import torchaudio
import keyboard





class str_enc():
    def __init__(self, string):
        self.name = string
    def __str__(self):
        return self.name




class Audio_Obj:
    def __init__(self):
        # Important paths
        self.model_fpath = 'saved_models/default/encoder.pt'
        self.synth_path = 'saved_models/default/synthesizer.pt'
        self.vocode_path = 'saved_models/default/vocoder.pt'

        # Synthesizer is initially NOne
        self.synthesizer = None

        # The embeddings are initially nothing
        self.embed = None

        # Utterances are the inputted audio clips
        self.utterances = set()






    # Load a sound file in
    # name - filename to load in
    # speaker_name - Path to file
    def load_from_browser(self, name, speaker_name):
        # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
        # playback, so as to have a fair comparison with the generated audio
        fpath = os.path.abspath(speaker_name+os.sep+name)
        wav = Synthesizer.load_preprocess_wav(fpath)
        print("Loaded %s" % name)

        self.add_real_utterance(wav, name, speaker_name)

    # Add an utterance
    def add_real_utterance(self, wav, name, speaker_name):
        # Compute the embedding
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # Save the current complete embedding
        self.embed = embed

        # Add the utterance
        spec = None # Spec not needed. Only needed for vis
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, False)
        self.utterances.add(utterance)

    # Initialize the encoder model to encode given audio clips
    def init_encoder(self):
        print("Loading the encoder %s... " % self.model_fpath)
        encoder.load_model(str_enc(self.model_fpath), device=torch.device("cpu"))
        print("Done")








    # This is where inferencing happens
    def synthesize(self, text, random_seed=None):
        # Make sure embeddings have been added
        assert self.embed is not None, "Make sure to add at least one embedding"

        print("Generating the mel spectrogram...")

        # Update the synthesizer random seed
        if random_seed != None:
            seed = int(random_seed)
        else:
            seed = None
        if seed is not None:
            torch.manual_seed(seed)

        # Synthesize the spectrogram
        if self.synthesizer is None or seed is not None:
            self.init_synthesizer()

        texts = text.split("\n")
        embeds = [self.embed] * len(texts)
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)

        # Save the currently generated utterance
        self.current_generated = ("", spec, breaks, None)


    # Initialize the synthesizer model to output audio
    # spectograms
    def init_synthesizer(self):
        model_fpath = str_enc(self.synth_path)

        print("Loading the synthesizer %s... " % model_fpath)
        self.synthesizer = Synthesizer(model_fpath)
        print("Done")






    # Used to do the second part of sysnthesis with the
    # synthesizer data
    # random_seed - Seed to initialize the model gneeration
    # traim_silences - True to trim excessive silences, False otherwise
    def vocode(self, random_seed=None, trim_silences=False):
        # Get the currently generated spectogram
        speaker_name, spec, breaks, _ = self.current_generated
        assert spec is not None

        # Initialize the vocoder model and make it determinstic, if user provides a seed
        if random_seed != None:
            seed = int(random_seed)
        else:
            seed = None
        if seed is not None:
            torch.manual_seed(seed)

        # Synthesize the waveform
        if not vocoder.is_loaded() or seed is not None:
            self.init_vocoder()

        # Generate a waveform from the spectogram
        def vocoder_progress(i, seq_len, b_size, gen_rate):
            real_time_factor = (gen_rate / Synthesizer.sample_rate) * 1000
            line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
                    % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
        if self.vocode_path is not None:
            wav = vocoder.infer_waveform(spec, progress_callback=vocoder_progress)
        else:
            print("Waveform generation with Griffin-Lim... ")
            wav = Synthesizer.griffin_lim(spec)
        print(" Done!", "append")

        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # Trim excessive silences
        if trim_silences == True:
            wav = encoder.preprocess_wav(wav)

        # Play the audio
        wav = torch.tensor(wav/np.abs(wav).max() * 0.97, dtype=torch.float32).unsqueeze(0)
        torchaudio.save("tmp.mp3", wav, Synthesizer.sample_rate)
        mixer.init()
        mixer.music.load('tmp.mp3')
        mixer.music.play()

        # # Compute the embedding
        # # TODO: this is problematic with different sampling rates, gotta fix it
        # if not encoder.is_loaded():
        #     self.init_encoder()
        # encoder_wav = encoder.preprocess_wav(wav)
        # embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # # Add the utterance
        # name = speaker_name + "_gen_%05d" % np.random.randint(100000)
        # utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, True)
        # self.utterances.add(utterance)

    # Play some audio
    def play(self, wav, sample_rate):
        try:
            sd.stop()
            sd.play(wav, sample_rate)
        except Exception as e:
            print(e)
            print("Error in audio playback. Try selecting a different audio output device.")
            print("Your device must be connected before you start the toolbox.")


    # Initialize the vocoder to generate new audio
    def init_vocoder(self):
        model_fpath = self.vocode_path
        # Case of Griffin-lim
        if model_fpath is None:
            return

        print("Loading the vocoder %s... " % model_fpath)
        vocoder.load_model(model_fpath)
        print("Done")







if __name__ == "__main__":
    # Create a new object
    obj = Audio_Obj()

    # Load in a file
    obj.load_from_browser("1.5.mp3", "data")
    obj.load_from_browser("2.5.mp3", "data")
    obj.load_from_browser("3.5.mp3", "data")
    obj.load_from_browser("4.5.mp3", "data")
    obj.load_from_browser("5.5.mp3", "data")
    obj.load_from_browser("6.5.mp3", "data")
    obj.load_from_browser("7.5.mp3", "data")
    obj.load_from_browser("8.5.mp3", "data")


    while True:
        # Get the text
        print("Prompt: ", end="")
        text = input()

        if text == "":
            break
        
        # Any punctuation is replaced with a newline
        text = text.replace(".", "\n").replace("?", "\n").replace("!", "\n")

        # Create the audio
        obj.synthesize(text)
        obj.vocode()