"""
Dexed VSTi audio renderer and presets database reader classes.

More information about the original DX7 paramaters:
https://www.chipple.net/dx7/english/edit.mode.html
https://djjondent.blogspot.com/2019/10/yamaha-dx7-algorithms.html
"""

import socket
import sys
import os
import pickle
import multiprocessing
import time
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


def get_partial_presets_df(db_row_index_limits):
    """ Returns a partial dataframe of presets from the DB, limited a tuple of row indexes
    (first and last included).

    Useful for fast DB reading, because it involves a lot of unpickling which can be parallelized. """
    db_path = PresetDatabase._get_db_path()
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    nb_rows = db_row_index_limits[1] - db_row_index_limits[0] + 1
    presets_df = pd.read_sql_query("SELECT * FROM preset LIMIT {} OFFSET {}"
                                   .format(nb_rows, db_row_index_limits[0]), conn)
    conn.close()
    return presets_df


class PresetDatabase:
    def __init__(self, num_workers=None):
        """ Opens the SQLite DB and copies all presets internally. This uses a lot of memory
        but allows easy multithreaded usage from multiple parallel dataloaders (1 db per dataloader). """
        self._db_path = self._get_db_path()
        conn = sqlite3.connect(self._db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        # We load the full presets table (full DB is usually a few dozens of megabytes)
        self.all_presets_df = self._load_presets_df_multiprocess(conn, cur, num_workers)
        # 20 megabytes for 30 000 presets
        self.presets_mat = self.all_presets_df['pickled_params_np_array'].values
        self.presets_mat = np.stack(self.presets_mat)
        # Memory save: param values are removed from the main dataframe
        self.all_presets_df.drop(columns='pickled_params_np_array', inplace=True)
        # Algorithms are also separately stored
        self._preset_algos = self.presets_mat[:, 4]
        self._preset_algos = np.asarray(np.round(1.0 + self._preset_algos * 31.0), dtype=np.int)
        # We also pre-load the names in order to close the sqlite DB
        names_df = pd.read_sql_query("SELECT * FROM param ORDER BY index_param", conn)
        self._param_names = names_df['name'].to_list()
        conn.close()

    def _load_presets_df_multiprocess(self, conn, cur, num_workers):
        if num_workers is None:
            num_workers = os.cpu_count() // 2
        cur.execute('SELECT COUNT(1) FROM preset')
        presets_count = cur.fetchall()[0][0]
        num_workers = np.minimum(presets_count, num_workers)
        # The last process might have a little more work to do
        rows_count_by_proc = presets_count // num_workers
        row_index_limits = list()
        for n in range(num_workers-1):
            row_index_limits.append([n * rows_count_by_proc, (n+1) * rows_count_by_proc - 1])
        # Last proc takes the remaining
        row_index_limits.append([(num_workers-1)*rows_count_by_proc, presets_count-1])
        with multiprocessing.Pool(num_workers) as p:
            partial_presets_dfs = p.map(get_partial_presets_df, row_index_limits)
        return pd.concat(partial_presets_dfs)

    @staticmethod
    def _get_db_path():
        return pathlib.Path(__file__).parent.joinpath('dexed_presets.sqlite')  # pkgutil would be better

    def __str__(self):
        return "{} DX7 presets in database '{}'.".format(len(self.all_presets_df), self._db_path)

    def get_nb_presets(self):
        return len(self.all_presets_df)

    def get_preset_name(self, idx):
        return self.all_presets_df.iloc[idx]['name']

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

    def get_preset_indexes_for_algorithms(self, algos):
        """ Returns a list of indexes of presets using the given algorithms in [[1 ; 32]] """
        indexes = []
        for i in range(self._preset_algos.shape[0]):
            if self._preset_algos[i] in algos:
                indexes.append(i)
        return indexes

    def get_size_info(self):
        """ Prints a detailed view of the size of this class and its main elements """
        main_df_size = self.all_presets_df.memory_usage(deep=True).values.sum()
        preset_values_size = self.presets_mat.size * self.presets_mat.itemsize
        return "Dexed Presets Database class size: " \
               "preset values matrix {:.1f} MB, presets dataframe {:.1f} MB"\
            .format(preset_values_size/(2**20), main_df_size/(2**20))

    @staticmethod
    def _get_presets_folder():
        return pathlib.Path(__file__).parent.absolute().joinpath('dexed_presets')

    def write_all_presets_to_files(self, verbose=True):
        """ Write all presets' parameter values to separate pickled files, for multi-processed multi-worker
        DataLoader. File names are presetXXXXXX_params.pickle where XXXXXX is the preset UID (it is not
        its row index in the SQLite database).

        Presets' names will be written to presetXXXXXX_name.txt,
        and comma-separated labels to presetXXXXXX_labels.txt.

        Performs consistency checks (e.g. labels, ...). TODO implement all consistency checks

        All files will be written to ./dexed_presets/ """
        presets_folder = self._get_presets_folder()
        if not os.path.exists(presets_folder):
            os.makedirs(presets_folder)
        for i in range(len(self.presets_mat)):
            preset_UID = self.all_presets_df.iloc[i]['index_preset']
            param_values = self.presets_mat[i, :]
            base_name = "preset{:06d}_".format(preset_UID)
            # ((un-)pickling has been done far too many times for these presets... could have been optimized)
            with open(presets_folder.joinpath(base_name + "params.pickle"), 'wb') as f:
                pickle.dump(param_values, f)
            with open(presets_folder.joinpath(base_name + "name.txt"), 'w') as f:
                f.write(self.all_presets_df.iloc[i]['name'])
            with open(presets_folder.joinpath(base_name + "labels.txt"), 'w') as f:
                labels = self.all_presets_df.iloc[i]['labels']
                labels_list = labels.split(',')
                for l in labels_list:
                    if not any([l == l_ for l_ in self.get_available_labels()]):  # Checks if any is True
                        raise ValueError("Label '{}' should not be available in self.all_presets_df".format(l))
                f.write(labels)
        if verbose:
            print("[dexed.PresetDatabase] Params, names and labels from SQLite DB written as .pickle and .txt files")

    @staticmethod
    def get_preset_params_values_from_file(preset_UID):
        return np.load(PresetDatabase._get_presets_folder()
                       .joinpath( "preset{:06d}_params.pickle".format(preset_UID)), allow_pickle=True)

    @staticmethod
    def get_preset_name_from_file(preset_UID):
        with open(PresetDatabase._get_presets_folder()
                  .joinpath( "preset{:06d}_name.txt".format(preset_UID)), 'r') as f:
            name = f.read()
        return name

    @staticmethod
    def get_available_labels():
        return 'harmonic', 'percussive', 'sfx'

    @staticmethod
    def get_preset_labels_from_file(preset_UID):
        """ Return the preset labels as a list of strings. """
        with open(PresetDatabase._get_presets_folder()
                  .joinpath("preset{:06d}_labels.txt".format(preset_UID)), 'r') as f:
            labels = f.read()
        return labels.split(',')


class Dexed:
    """ A Dexed (DX7) synth that can be used through RenderMan for offline wav rendering. """

    def __init__(self, plugin_path="/home/gwendal/Jupyter/AudioPlugins/Dexed.so",
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
        return "Plugin loaded from {}, Fs={}Hz, buffer {} samples."\
               "MIDI note on duration: {:.1f}s / {:.1f}s total."\
            .format(self.plugin_path, self.Fs, self.buffer_size,
                    self.midi_note_duration_s, self.render_duration_s)

    def render_note(self, midi_note, midi_velocity, normalize=False):
        """ Renders a midi note (for the currently set patch) and returns the normalized float array. """
        self.engine.render_patch(midi_note, midi_velocity, self.midi_note_duration_s, self.render_duration_s)
        audio_out = self.engine.get_audio_frames()
        audio = np.asarray(audio_out)
        fadeout_len = int(np.floor(self.Fs * self.fadeout_duration_s))
        if fadeout_len > 1:  # fadeout might be disabled if too short
            fadeout = np.linspace(1.0, 0.0, fadeout_len)
            audio[-fadeout_len:] = audio[-fadeout_len:] * fadeout
        if normalize:
            return audio / np.abs(audio).max()
        else:
            return audio

    def render_note_to_file(self, midi_note, midi_velocity, filename="./dexed_output.wav"):
        """ Renders a midi note (for the currently set patch), normalizes it and stores it
        to a 16-bit PCM wav file. """
        assert False  # deprecated function
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
        assert False  # deprecated - should return the modified params as well
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
        """ Internally sets the modified preset, and returns the list of parameter values. """
        assert self.current_preset is not None
        self.current_preset[0] = (0, 1.0)  # filter cutoff
        self.current_preset[1] = (1, 0.0)  # filter reso
        self.current_preset[2] = (2, 1.0)  # output vol
        self.current_preset[3] = (3, 0.5)  # master tune
        self.current_preset[13] = (13, 0.5)  # Sets the 'middle-C' note to the default C3 value
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    @staticmethod
    def set_default_general_filter_and_tune_params_(preset_params):
        """ Modifies some params in-place for the given numpy array """
        preset_params[[0, 1, 2, 3, 13]] = np.asarray([1.0, 0.0, 1.0, 0.5, 0.5])

    def set_all_oscillators_on(self):
        """ Internally sets the modified preset, and returns the list of parameter values. """
        assert self.current_preset is not None
        for idx in [44, 66, 88, 110, 132, 154]:
            self.current_preset[idx] = (idx, 1.0)
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    @staticmethod
    def set_all_oscillators_on_(preset_params):
        """ Modifies some params of the given numpy array to ensure that all operators (oscillators) are ON.
        Data is modified in place. """
        preset_params[[44, 66, 88, 110, 132, 154]] = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    @staticmethod
    def set_all_oscillators_off_(preset_params):
        """ Modifies some params of the given numpy array to ensure that all operators (oscillators) are OFF.
        Data is modified in place. """
        preset_params[[44, 66, 88, 110, 132, 154]] = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @staticmethod
    def set_oscillators_on_(preset_params, operators_to_turn_on):
        """ Modifies some params of the given numpy array to turn some operators ON. Data is modified in place.

        :param preset_params: Numpy Array of preset parameters values.
        :param operators_to_turn_on: List of integers in [1, 6]
        """
        Dexed.set_all_oscillators_off_(preset_params)
        for op_number in operators_to_turn_on:
            preset_params[44 + 22 * (op_number-1)] = 1.0

    def prevent_SH_LFO(self):
        """ If the LFO Wave is random S&H, transforms it into a square LFO wave to get deterministic
        results. Internally sets the modified preset, and returns the list of parameter values.  """
        if self.current_preset[12][1] > 0.95:  # S&H wave corresponds to a 1.0 param value
            self.current_preset[12] = (12, 4.0 / 5.0)  # Square wave is number 4/6
        self.engine.set_patch(self.current_preset)
        return [v for _, v in self.current_preset]

    @staticmethod
    def prevent_SH_LFO_(preset_params):
        """ Modifies some params in-place for the given numpy array """
        if preset_params[12] > 0.95:
            preset_params[12] = 4.0 / 5.0

    @staticmethod
    def get_midi_key_related_param_indexes():
        """ Returns a list of indexes of all DX7 parameters that apply a modulation depending on the MIDI key
        (note and/or velocity). These will be very hard to learn without providing multiple-notes input
        to the encoding network. """
        # (6. 'OSC KEY SYNC' (LFO) does NOT depend on the midi note (it syncs or not LFO phase on midi event).)
        # All the KEY L/R stuff (with breakpoint at some MIDI note) effects are really dependant on the MIDI key.
        # 36. breakpoint.
        # 37/38: L/R scale (curve) depth.
        # 39/40: L/R scale (=curve) type: +/-lin or +/-exp.
        return [(36 + 22*i) for i in range(6)]\
            + [(37 + 22*i) for i in range(6)] + [(38 + 22*i) for i in range(6)]\
            + [(39 + 22*i) for i in range(6)] + [(40 + 22*i) for i in range(6)] \
            + [(43 + 22 * i)for i in range(6)]

    @staticmethod
    def get_mod_wheel_related_param_indexes():
        """ Returns a list of indexes of all DX7 parameters that influence sound depending on the MIDI
        mod wheel. These should always be learned because they are also related to LFO modulation
        (see https://fr.yamaha.com/files/download/other_assets/9/333979/DX7E1.pdf page 26) """
        # OPx A MOD SENS + Pitch general mod sens
        return [(42 + 22*i) for i in range(6)] + [14]

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
        elif param_index == 14:  # pitch modulation sensitivity
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
            elif (param_index % 22) == (42 % 22):  # OPx A modulation sensitivity
                return 4
            elif (param_index % 22) == (43 % 22):  # OPx Key Velocity
                return 8
            elif (param_index % 22) == (44 % 22):  # OPx Switch (off/on)
                return 2
            else:  # all other are considered non-discrete  # TODO return 100
                return -1
        else:  # all other are considered non-discrete
            return -1

    @staticmethod
    def get_numerical_params_indexes():
        indexes = [0, 1, 2, 3, 5,  # cutoff, reso, output, master tune, feedback (card:8)
                   7, 8, 9, 10,  # lfo speed, lfo delay (before LFO actually modulates), lfo pm depth, lfo am depth
                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # transpose, pitch mod sensitivity, pitch EG rates/levels
        for i in range(6):  # operators
            for j in [23, 24, 25, 26, 27, 28, 29, 30]:  # rates and levels
                indexes.append(j + 22*i)
            indexes.append(31 + 22*i)  # output level
            indexes.append(33 + 22*i)  # freq coarse
            indexes.append(34 + 22*i)  # freq fine
            indexes.append(35 + 22*i)  # detune (these 3 parameters kind of overlap...)
            indexes.append(36 + 22*i)  # L/R scales breakpoint
            indexes.append(37 + 22*i)  # L scale depth
            indexes.append(38 + 22*i)  # R scale depth
            indexes.append(41 + 22*i)  # rate scaling (card:8)
            indexes.append(42 + 22*i)  # amplitude mod sensitivity (card:4)
            indexes.append(43 + 22*i)  # key velocity (card:8)
        return indexes

    @staticmethod
    def get_categorical_params_indexes():
        indexes = [4, 6, 11, 12]  # algorithm, osc key sync, lfo key sync, lfo wave
        for i in range(6):  # operators
            indexes.append(32 + 22*i)  # mode (ratio or fixed frequency)
            indexes.append(39 + 22*i)  # L scale
            indexes.append(40 + 22*i)  # R scale
            indexes.append(44 + 22*i)  # op on/off switch
        return indexes


if __name__ == "__main__":

    print("Machine: '{}' ({} CPUs)".format(socket.gethostname(), os.cpu_count()))

    t0 = time.time()
    dexed_db = PresetDatabase()
    print("{} (loaded in {:.1f}s)".format(dexed_db, time.time() - t0))
    names = dexed_db.get_param_names()
    #print("Labels example: {}".format(dexed_db.get_preset_labels_from_file(3)))

    print("numerical VSTi params: {}".format(Dexed.get_numerical_params_indexes()))
    print("categorical VSTi params: {}".format(Dexed.get_categorical_params_indexes()))

    if False:
        # ***** RE-WRITE ALL PRESETS TO SEPARATE PICKLE/TXT FILES *****
        # Approx. 360Mo (yep, the SQLite DB is much lighter...) for all params values + names + labels
        dexed_db.write_all_presets_to_files()

    if False:
        # Test de lecture des fichiers pickled - pas besoin de la DB lue en entier
        preset_values = PresetDatabase.get_preset_params_values_from_file(0)
        preset_name = PresetDatabase.get_preset_name_from_file(0)
        print(preset_name)

    if False:
        # Test du synth lui-même
        dexed = Dexed()
        print(dexed)
        print("Plugin params: ")
        print(dexed.engine.get_plugin_parameters_description())

        #dexed.assign_random_preset_short_release()
        #pres = dexed.preset_db.get_preset_values(0, plugin_format=True)
        #dexed.assign_preset_from_db(100)
        #print(dexed.current_preset)

        #dexed.render_note(57, 100, filename="Test.wav")

        print("{} presets use algo 5".format(len(dexed_db.get_preset_indexes_for_algorithm(5))))


