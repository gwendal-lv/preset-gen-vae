
import socket
import sys
import os
import numpy as np
from scipy.io import wavfile
import sqlite3
import io
import pandas as pd

# DB reading from the package itself
import pathlib
#import pkgutil

import librenderman as rm  # A symbolic link to the actual librenderman.so must be found in the current folder


# Pickled numpy arrays storage in sqlite3 DB
def adapt_array(arr):
    """ http://stackoverflow.com/a/31312102/190597 (SoulNibbler) """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter("NPARRAY", convert_array)


class PresetDatabase:
    def __init__(self):
        """ Opens the SQLite DB and copies all presets internally. This uses more memory
        but allows easy multithreaded usage from multiple parallel dataloaders (1 db per dataloader). """
        self._db_path = pathlib.Path(__file__).parent.joinpath('dexed_presets.sqlite')  # pkgutil would be better
        conn = sqlite3.connect(self._db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        # We load the full presets table (full DB is usually a few dozens of megabytes)
        self._all_presets_df = pd.read_sql_query("SELECT * FROM preset", conn)
        # 20 megabytes for 30 000 presets
        self.presets_mat = [self._all_presets_df.iloc[i]['pickled_params_np_array']
                            for i in range(len(self._all_presets_df))]
        self.presets_mat = np.asarray(self.presets_mat, dtype=np.float32)
        # Memory save: param values are removed from the main dataframe
        self._all_presets_df.drop(columns='pickled_params_np_array', inplace=True)
        # Algorithms are also separately stored
        self._preset_algos = self.presets_mat[:, 4]
        self._preset_algos = np.asarray(np.round(1.0 + self._preset_algos * 31.0), dtype=np.int)
        # We also pre-load the names in order to close the sqlite DB
        names_df = pd.read_sql_query("SELECT * FROM param ORDER BY index_param", conn)
        self._param_names = names_df['name'].to_list()
        conn.close()

    def __str__(self):
        return "{} DX7 presets in database '{}'.".format(len(self._all_presets_df), self._db_path)

    def get_nb_presets(self):
        return len(self._all_presets_df)

    def get_preset_name(self, idx):
        return self._all_presets_df.iloc[idx]['name']

    def get_preset_values(self, idx, plugin_format=False):
        """ Returns a preset from the DB.

        :param idx: the preset 'row line' in the DB (not the index_preset value, which is an ID)
        :param plugin_format: if True, returns a list of (param_index, param_value) tuples. If False, returns the
            numpy array of param values. """
        preset_values = self.presets_mat[idx, :]
        if plugin_format:
            return self.get_params_in_plugin_format(preset_values)
        else:
            return preset_values

    @staticmethod
    def get_params_in_plugin_format(params):
        """ Converts a 1D array of param values into an list of (idx, param_value) tuples """
        preset_values = np.asarray(params, dtype=np.double)  # np.float32 is not valid for RenderMan
        # Dexed parameters are nicely ordered from 0 to 154
        return [(i, preset_values[i]) for i in range(preset_values.shape[0])]

    def get_param_names(self):
        return self._param_names

    def get_preset_indexes_for_algorithm(self, algo):
        """ Returns a list of indexes of presets using the given algorithm in [[1 ; 32]] """
        indexes = []
        for i in range(self._preset_algos.shape[0]):
            if self._preset_algos[i] == algo:
                indexes.append(i)
        return indexes

    def get_size_info(self):
        """ Prints a detailed view of the size of this class and its main elements """
        main_df_size = self._all_presets_df.memory_usage(deep=True).values.sum()
        preset_values_size = self.presets_mat.size * self.presets_mat.itemsize
        return "Dexed Presets Database class size: " \
               "preset values matrix {:.1f} MB, presets dataframe {:.1f} MB"\
            .format(preset_values_size/(2**20), main_df_size/(2**20))


class Dexed:
    """ A Dexed (DX7) synth that can be used through RenderMan for offline wav rendering. """

    def __init__(self, plugin_path="/home/gwendal/Jupyter/AudioPlugins/Dexed.so",
                 midi_note_duration_s=4.0, render_duration_s=5.0,
                 sample_rate=22050,  # librosa default sr
                 buffer_size=512, fft_size=512):
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
        return "Plugin loaded from {}, Fs={}Hz, buffer {} samples."\
               "MIDI note on duration: {:.1f}s / {:.1f}s total. {}"\
            .format(self.plugin_path, self.Fs, self.buffer_size,
                    self.midi_note_duration_s, self.render_duration_s)

    def render_note(self, midi_note, midi_velocity):
        """ Renders a midi note (for the currently set patch) and returns the normalized float array. """
        self.engine.render_patch(midi_note, midi_velocity, self.midi_note_duration_s, self.render_duration_s)
        audio_out = self.engine.get_audio_frames()
        audio = np.asarray(audio_out)
        return audio / np.abs(audio).max()

    def render_note_to_file(self, midi_note, midi_velocity, filename="./dexed_output.wav"):
        """ Renders a midi note (for the currently set patch), normalizes it and stores it
        to a 16-bit PCM wav file. """
        self.engine.render_patch(midi_note, midi_velocity, self.midi_note_duration_s, self.render_duration_s)
        # RenderMan wav writing is broken - using scipy instead
        audio_out = self.engine.get_audio_frames()
        audio = np.asarray(audio_out)
        max_amplitude = np.abs(audio).max()
        audio = ((2**15 - 1) / max_amplitude) * audio
        audio = np.array(np.round(audio), dtype=np.int16)
        wavfile.write(filename, self.Fs, audio)

    def assign_preset(self, preset):
        """ :param preset: List of tuples (param_idx, param_value) """
        self.current_preset = preset
        self.engine.set_patch(self.current_preset)

    def assign_random_preset_short_release(self):
        """ Generates a random preset with a short release time - to ensure a limited-duration
         audio recording, and configures the rendering engine to use that preset. """
        self.current_preset = dexed.preset_gen.get_random_patch()
        self.set_release_short()
        self.engine.set_patch(self.current_preset)

    def set_release_short(self, eg_4_rate_min=0.5):
        for i, param in enumerate(self.current_preset):
            idx, value = param  # a param is actually a tuple...
            # Envelope release level: always to zero (or would be an actual hanging note)
            if idx == 30 or idx == 52 or idx == 74 or idx == 96 or idx == 118 or idx == 140:
                self.current_preset[i] = (idx, 0.0)
            # Envelope release time: quite short (higher float value: shorter release)
            elif idx == 26 or idx == 48 or idx == 70 or idx == 92 or idx == 114 or idx == 136:
                self.current_preset[i] = (idx, max(eg_4_rate_min, value))
        self.engine.set_patch(self.current_preset)

    def set_default_general_filter_and_tune_params(self):
        assert self.current_preset is not None
        self.current_preset[0] = (0, 1.0)  # filter cutoff
        self.current_preset[1] = (1, 0.0)  # filter reso
        self.current_preset[2] = (2, 1.0)  # output vol
        self.current_preset[3] = (3, 0.5)  # master tune
        self.current_preset[13] = (13, 0.5)  # Sets the 'middle-C' note to the default C3 value
        self.engine.set_patch(self.current_preset)

    def prevent_SH_LFO(self):
        """ If the LFO Wave is random S&H, transforms it into a square LFO wave to get deterministic results. """
        if self.current_preset[12][1] > 0.95:  # S&H wave corresponds to a 1.0 param value
            self.current_preset[12] = (12, 4.0 / 5.0)  # Square wave is number 4/6
        self.engine.set_patch(self.current_preset)

    @staticmethod
    def get_param_cardinality(param_index):
        """ Returns the number of possible values for a given parameter, or -1 if the param
        is considered continuous (100 discrete values). """
        if param_index == 4:  # Algorithm
            return 32
        elif param_index == 5:  # Feedback
            return 8
        elif param_index == 6:  # OSC key sync (off/on)
            return 2
        elif param_index == 11:  # LFO key sync (off/on)
            return 2
        elif param_index == 12:  # LFO wave
            return 6
        elif param_index == 14:  # pitch mode sensitivity
            return 8
        elif param_index >= 23:  # oscillators (operators) params
            if (param_index % 22) == (32 % 22):  # OPx Mode (ratio/fixed)
                return 2
            elif (param_index % 22) == (33 % 22):  # OPx F coarse
                return 32
            elif (param_index % 22) == (35 % 22):  # OPx OSC Detune
                return 15
            elif (param_index % 22) == (39 % 22):  # OPx L Key Scale (-lin, -exp, +exp, +lin)
                return 4
            elif (param_index % 22) == (40 % 22):  # OPx R Key Scale (-lin, -exp, +exp, +lin)
                return 4
            elif (param_index % 22) == (41 % 22):  # OPx Rate Scaling
                return 8
            elif (param_index % 22) == (42 % 22):  # OPx A (amplitude?) mode sensitivity
                return 4
            elif (param_index % 22) == (43 % 22):  # OPx Key Velocity
                return 8
            elif (param_index % 22) == (44 % 22):  # OPx Switch (off/on)
                return 2
            else:  # all other are considered non-discrete
                return -1
        else:  # all other are considered non-discrete
            return -1


if __name__ == "__main__":

    print("Machine: '{}' ({} CPUs)".format(socket.gethostname(), os.cpu_count()))

    dexed = Dexed()
    print(dexed)
    print("Plugin params: ")
    print(dexed.engine.get_plugin_parameters_description())

    #dexed.assign_random_preset_short_release()
    #pres = dexed.preset_db.get_preset_values(0, plugin_format=True)
    #dexed.assign_preset_from_db(100)
    #print(dexed.current_preset)

    #dexed.render_note(57, 100, filename="Test.wav")

    names = dexed.preset_db.get_param_names()

    print("{} presets use algo 5".format(len(dexed.preset_db.get_preset_indexes_for_algorithm(5))))

    pass


