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
        self.param_names     = pickle.load(open("/home/irisib/Bureau/nn-synth-interp/synth/diva_presets/diva_params.pkl", "rb"))
        self.all_presets     = pickle.load(open("/home/irisib/Bureau/nn-synth-interp/synth/diva_presets/diva_presets.pkl", "rb"))
        self.all_presets_raw = pickle.load(open("/home/irisib/Bureau/nn-synth-interp/synth/diva_presets/diva_presets_raw.pkl", "rb"))
        self.db_path         = ("/home/irisib/Bureau/nn-synth-interp/synth/diva_presets.pkl")

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

    def get_preset_raw_format(self, idx, plugin_format=True):
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

    def write_all_presets_to_files(self, format=True):
        """ Write all presets' parameter values to separate pickled files, for multi-processed multi-worker
        DataLoader. File names are presetXXXXXX_params.pickle where XXXXXX is the preset UID (it is not
        its row index)"""
        tuple_midi = []
        midi = []
        if format is True:
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
        else:
            parameters = pickle.load(open("diva_presets/diva_params.pkl", "rb"))
            with codecs.open("diva_presets.json", encoding="utf-8") as js:
                diva_dataset = json.load(js)
                for k_hash, element in diva_dataset.items():
                    for parameter in parameters:
                        midi.append(element['MIDI'][parameter])
                    tuple_midi.append(list(midi))
                    midi.clear()
            new_file = open("diva_presets/diva_presets_raw.pkl", "wb")
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
        self.current_preset[13] = (13, 0.5)  # transpose
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    def set_dry_params_off(self):
        """ Internally sets the modified preset, and returns the list of parameter values. """
        for dry_param in [195, 205, 234, 244]:
            self.current_preset[dry_param] = (dry_param, 1.0)
        for wet_param in [182, 189, 196, 221, 228, 235]:
            self.current_preset[wet_param] = (wet_param, 0.0)
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    def set_osc_params_off(self, osc1=False, osc2=False, osc3 = False):
        """ Internally sets the modified preset, and returns the list of parameter values. """
        if osc1 == False and osc2 == False and osc3 == False:
            for osc_param in list(range(85, 138)):
                self.current_preset[osc_param] = (osc_param, 0.0)
        else:
            if not osc1:
                for osc_param in [86, 97]:
                    self.current_preset[osc_param] = (osc_param, 0.0)
            if not osc2:
                for osc_param in [87, 98]:
                    self.current_preset[osc_param] = (osc_param, 0.0)
            if not osc3:
                for osc_param in [88, 99]:
                    self.current_preset[osc_param] = (osc_param, 0.0)
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    def set_scope_params_off(self):
        """ Internally sets the modified preset, and returns the list of parameter values. """
        for scope_param in [176, 177]:
                self.current_preset[scope_param] = (scope_param, 0.0)
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    def set_fx_params_off(self):
        """ Internally sets the modified preset, and returns the list of parameter values. """
        for fx_param in [168, 200, 201, 239, 240, 263, 274, 275, 276, 278]:
                self.current_preset[fx_param] = (fx_param, 0.0)
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

    @staticmethod
    def get_numerical_params_indexes():
        indexes = [0, 3, 8, 9, 10, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 43, 44,
                   45, 46, 47, 48, 54, 58, 59, 61, 62, 64, 68, 69, 71, 72, 74, 75, 76, 86, 87, 88, 89, 90, 91, 92, 93,
                   94, 96, 97, 98, 99, 104, 106, 108, 110, 122, 131, 132, 134, 136, 138, 140, 141, 143, 145, 148, 149,
                   151, 153, 154, 155, 160, 162, 164, 166, 167, 168, 171, 173, 175, 176, 177, 180, 181, 182, 184, 185,
                   186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
                   208, 209, 210, 211, 212, 213, 214, 215, 219, 220, 221, 223, 224, 225, 227, 228, 229, 230, 231, 232,
                   233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 247, 248, 249, 250, 251, 252, 253,
                   254, 256, 258, 264, 265, 266, 267, 269, 274, 275, 276, 277]
        return indexes

    @staticmethod
    def get_categorical_params_indexes():
        indexes = [1, 2, 4, 5, 6, 7, 11, 12, 13, 14, 16, 17, 18, 19, 38, 39, 40, 41, 42, 49, 50, 51, 52, 53, 55, 56, 57,
                   60, 63, 65, 66, 67, 70, 73, 77, 78, 79, 80, 81, 82, 83, 84, 85, 95, 100, 101, 102, 103, 105, 107,
                   109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126, 127, 128, 129,
                   130, 133, 135, 137, 139, 142, 144, 146, 147, 150, 152, 156, 157, 158, 159, 161, 163, 165, 169,
                   170, 172, 174, 178, 179, 183, 187, 207, 216, 217, 218, 222, 226, 246, 255, 257, 259, 260, 261,
                   262, 263, 268, 270, 271, 272, 273, 278, 279, 280]
        return indexes


if __name__ == "__main__":

    print("Machine: '{}' ({} CPUs)".format(socket.gethostname(), os.cpu_count()))


    t0 = time.time()
    diva_db = PresetDatabase()
    print(diva_db._get_presets_folder())
    print(diva_db.get_params_in_plugin_format(self = diva_db,params= diva_db.get_preset_values(555, plugin_format=False)))
    diva_db.get_size_info()
    print("{} (loaded in {:.1f}s)".format(diva_db, time.time() - t0))
    names = diva_db.get_param_names()

    diva_db.write_all_presets_to_files(format=False)
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

    print("\n============== CURRENT PRESET ==============\n")
    print(diva.current_preset)

    #diva.render_note_to_file(57, 100, 5) TODO
    audio = diva.render_note(57, 100)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(audio)
    plt.subplot(212)
    plt.specgram(audio, NFFT=diva.fft_size, Fs=diva.Fs, noverlap=256)
    plt.show()
    #print("{} presets use algo 5".format(len(diva_db.get_preset_indexes_for_algorithm(5))))

