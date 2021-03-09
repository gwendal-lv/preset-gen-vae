import os
import pathlib
import socket
import time
import sys
import matplotlib.pyplot as plt
import librenderman as rm
import json
import json, codecs
import numpy as np
import scipy
import librosa
import soundfile as sf
from matplotlib import cm
import pickle

class PresetDatabase:
    def __init__(self, num_workers=None):
        self.param_names = pickle.load(open("/home/irisib/Bureau/nn-synth-interp/synth/diva_presets/diva_params.pkl", "rb"))
        self.all_presets = pickle.load(open("/home/irisib/Bureau/nn-synth-interp/synth/diva_presets/diva_presets.pkl", "rb"))
        self.db_path = ("/home/irisib/Bureau/nn-synth-interp/synth/diva_presets.pkl")

    @staticmethod
    def get_db_path():
        return pathlib.Path(__file__).parent.joinpath('diva_presets.json')

    def __str__(self):
        return "{} Diva presets in database '{}'.".format(len(self.all_presets), self.db_path)

    def get_nb_presets(self):
        return len(self.all_presets)

    def get_nb_params(self):
        return len(self.param_names)

    def get_preset_values(self, idx, plugin_format=True):
        """ Returns a preset from the DB.
        :param idx: index of preset
        :param plugin_format: if True, returns a list of (param_index, param_value) tuples. If False, returns the
            numpy array of param values. """
        if plugin_format:
            return self.all_presets[idx]
        else:
            preset_values = []
            for element in self.all_presets[idx]:
                preset_values.append(element[1])
            return preset_values

    @staticmethod
    def get_params_in_plugin_format(self, params):
        """ Converts a 1D array of param values into an list of (idx, param_value) tuples """
        index_midi = list(range(len(self.param_names)))
        return list(zip(index_midi, params))

    def get_param_names(self):
        return self.param_names

    def get_size_info(self):
        """ Prints a detailed view of the size of this class and its main elements """
        print("All params memory usage  : " + str(sys.getsizeof(self.param_names)/(2**20)) + " MB")
        print("All presets memory usage : " + str(sys.getsizeof(self.all_presets)/(2**20)) + " MB")

    @staticmethod
    def _get_presets_folder():
        return pathlib.Path(__file__).parent.absolute().joinpath('diva_presets')

    def write_all_presets_to_files(self, verbose=True): # QUESTION
        """ Write all presets' parameter values to separate pickled files, for multi-processed multi-worker
        DataLoader. File names are presetXXXXXX_params.pickle where XXXXXX is the preset UID (it is not
        its row index)"""

        tuple_midi = []
        midi = []
        parameters = pickle.load(open("diva_presets/diva_params.pkl", "rb"))
        index_midi = list(range(len(self.param_names)))
        with codecs.open("diva_presets.json", encoding="utf-8") as js:
            diva_dataset = json.load(js)
            for k_hash, element in diva_dataset.items():
                for parameter in parameters:
                    midi.append(element['MIDI'][parameter])
                tuple_midi.append(list(zip(index_midi, midi)))
                midi.clear()
        new_file = open("diva_presets/diva_presets.pkl", "wb")
        pickle.dump(tuple_midi, new_file)
        new_file.close()

class Diva:

    """ A Diva synth that can be used through RenderMan for offline wav rendering. """
    def __init__(self, plugin_path="/home/irisib/Bureau/nn-synth-interp/synth/diva_presets/Diva64.so",
                midi_note_duration_s=3.0, render_duration_s=4.0,
                sample_rate=22050,  # librosa default sr
                buffer_size=512, fft_size=512,
                fadeout_duration_s=0.1):
        self.fadeout_duration_s = fadeout_duration_s  # To reduce STFT discontinuities with long-release presets
        self.midi_note_duration_s = midi_note_duration_s
        self.render_duration_s = render_duration_s

        self.plugin_path = plugin_path
        self.Fs = sample_rate
        self.buffer_size = buffer_size
        self.fft_size = fft_size  # FFT not used

        self.engine = rm.RenderEngine(self.Fs, self.buffer_size, self.fft_size)
        self.engine.load_plugin(self.plugin_path)

        # A generator preset is a list of (int, float) tuples.
        self.preset_gen = rm.PatchGenerator(self.engine)  # 'RenderMan' generator
        self.current_preset = None

    def __str__(self):
        return "Plugin loaded from {}, Fs={}Hz, buffer {} samples." \
                "MIDI note on duration: {:.1f}s / {:.1f}s total." \
            .format(self.plugin_path, self.Fs, self.buffer_size,
                    self.midi_note_duration_s, self.render_duration_s)

    def render_note(self, midi_note, midi_velocity, normalize=False):
        """ Renders a midi note (for the currently set patch) and returns the normalized float array. """
        self.engine.render_patch(midi_note, midi_velocity, self.midi_note_duration_s, self.render_duration_s)
        audio = self.engine.get_audio_frames()
        fadeout_len = int(np.floor(self.Fs * self.fadeout_duration_s))
        if fadeout_len > 1:  # fadeout might be disabled if too short
            fadeout = np.linspace(1.0, 0.0, fadeout_len)
            audio[-fadeout_len:] = audio[-fadeout_len:] * fadeout
        if normalize:
            return audio / np.abs(audio).max()
        else:
            return audio

    def render_note_to_file(self, midi_note, midi_velocity, index_file):
        """ Renders a midi note (for the currently set patch), normalizes it and stores it
        to a 16-bit PCM wav file. """
        self.engine.render_patch(midi_note, midi_velocity, self.midi_note_duration_s, self.render_duration_s)
        audio = self.engine.get_audio_frames()
        filename = ('Rendu/sound' + str(index_file) + '.wav')
        sf.write(filename, audio, self.Fs)

    def assign_preset(self, preset):
        """ :param preset: List of tuples (param_idx, param_value) """
        self.current_preset = preset
        self.engine.set_patch(self.current_preset)

    def assign_random_preset(self):
        """ Generates a random preset with a short release time - to ensure a limited-duration
         audio recording, and configures the rendering engine to use that preset. """
        self.current_preset = diva.preset_gen.get_random_patch()
        self.engine.set_patch(self.current_preset)

    def set_default_general_filter_and_tune_params(self):
        """ Internally sets the modified preset, and returns the list of parameter values. """
        assert self.current_preset is not None
        self.current_preset[0] = (0, 1.0)  # output main
        self.current_preset[13] = (13, 1.0)  # output main
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    @staticmethod
    def get_param_cardinality(param_index):
        """ Returns the number of possible values for a given parameter, or -1 if the param
        is considered continuous (100 discrete values). """
        index = {
            1: 2,
            2: 2,
            4: 8,
            5: 6,
            6: 5,
            7: 2,
            11: 27,
            12: 27,
            13: 2,
            14: 49,
            16: 3,
            17: 2,
            18: 4,
            19: 2,
            38: 3,
            39: 4,
            40: 2,
            41: 2,
            42: 2,
            49: 3,
            50: 5,
            51: 2,
            52: 2,
            53: 2,
            55: 27,
            56: 4,
            57: 8,
            60: 24,
            63: 24,
            65: 27,
            66: 4,
            67: 8,
            70: 24,
            73: 24,
            77: 24,
            78: 24,
            79: 24,
            80: 24,
            81: 24,
            82: 24,
            83: 24,
            84: 24,
            85: 5,
            95: 2,
            100: 4,
            101: 6,
            102: 6,
            103: 24,
            105: 24,
            107: 24,
            109: 24,
            111: 2,
            112: 2,
            113: 2,
            114: 2,
            115: 2,
            116: 2,
            117: 2,
            118: 2,
            119: 2,
            120: 3,
            121: 2,
            123: 2,
            124: 2,
            125: 2,
            126: 2,
            127: 2,
            128: 3,
            129: 2,
            130: 4,
            133: 2,
            135: 24,
            137: 24,
            139: 4,
            142: 2,
            144: 24,
            146: 5,
            147: 5,
            150: 24,
            152: 24,
            156: 2,
            157: 2,
            158: 2,
            159: 4,
            161: 24,
            163: 24,
            165: 24,
            169: 2,
            170: 24,
            172: 24,
            174: 3,
            178: 5,
            179: 3,
            183: 2,
            187: 2,
            207: 3,
            216: 4,
            217: 5,
            218: 3,
            222: 2,
            226: 2,
            246: 3,
            255: 4,
            257: 4,
            259: 6,
            260: 4,
            261: 5,
            262: 13,
            263: 2,
            268: 24,
            270: 2,
            271: 4,
            272: 2,
            273: 2,
            278: 7,
            279: 7,
            280: 2,
        }
        return index.get(param_index, -1)


if __name__ == "__main__":

    print("Machine: '{}' ({} CPUs)".format(socket.gethostname(), os.cpu_count()))


    t0 = time.time()
    diva_db = PresetDatabase()
    print(diva_db._get_presets_folder())
    print(diva_db.get_params_in_plugin_format(self = diva_db,params= diva_db.get_preset_values(555, plugin_format=False)))
    diva_db.get_size_info()
    print("{} (loaded in {:.1f}s)".format(diva_db, time.time() - t0))
    names = diva_db.get_param_names()
    #print("Labels example: {}".format(dexed_db.get_preset_labels_from_file(3)))
    #preset_values = PresetDatabase.get_preset_params_values_from_file(0)
    param_names = PresetDatabase.get_param_names(diva_db)
    print("==============PARAM NAMES==============")
    print(param_names)
    print("==============NUM OF PRESETS==============")
    print(diva_db.get_nb_presets())

    patchTest = diva_db.get_preset_values(555)
    print("==============PATCH TEST==============")
    print(patchTest)
    diva = Diva()
    print("==============DIVA==============")
    print(diva)
    #diva.assign_random_preset()
    diva.assign_preset(patchTest)
    print("\n============== PATCH TEST VALUES NORMALIZED ==============\n")
    print(diva.set_default_general_filter_and_tune_params())
    print("\n============== PARAM DESC ==============\n")
    #print(diva.engine.get_plugin_parameters_description())

    #pres = dexed.preset_db.get_preset_values(0, plugin_format=True)
    #dexed.assign_preset_from_db(100)
    print("\n============== CURRENT PRESET ==============\n")
    print(diva.current_preset)

    diva.render_note_to_file(57, 100, 5)
    audio = diva.render_note(57, 100)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(audio)
    plt.subplot(212)
    plt.specgram(audio, NFFT=diva.fft_size, Fs=diva.Fs, noverlap=256)
    plt.show()
    #print("{} presets use algo 5".format(len(diva_db.get_preset_indexes_for_algorithm(5))))

