---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: splash
classes: wide

---

<link rel="stylesheet" href="assets/css/styles.css">

Accompanying material for the DAFx21 submission "Improving Synthesizer Programming from Variational Autoencoders Latent Space" (under review)

Authors: G. Le Vaillant, T. Dutoit, S. Dekeyser

<img src="assets/images/GitHub-mark-64px.png" style="width: 24px"> [gwendal-lv/preset-gen-vae](https://github.com/gwendal-lv/preset-gen-vae)

---

# Automatic synthesizer programming

These first examples demonstrate how our model can be used to program a [Dexed FM synthesizer](https://github.com/asb2m10/dexed) (Yamaha DX7 clone) to reproduce a given audio input.
Input sounds were generated using original presets from the held-out test set of our [30k presets dataset](https://github.com/gwendal-lv/preset-gen-vae/blob/main/synth/dexed_presets.sqlite).
Each preset contains 144 learnable parameters.


<div class="figure">
    <table>
        <tr>
            <th></th>
            <th>Original preset</th>
            <th colspan=3 style="text-align: center">Inferred presets <sup>*</sup> </th>
        </tr>
        <tr style="text-align: center">
            <td></td>
            <td></td>
            <td colspan=3>Parameters learnable representation:</td>
        </tr>
        <tr style="text-align: center">
            <td></td>
            <td></td>
            <td>Num only</td>
            <td>NumCat</td>
            <td>NumCat++</td>
        </tr>
        <tr>
            <td>"Rhodes 3"<br/>(harmonic)</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/021894_GT_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/021894_GT_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/021894_numonly_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/021894_numonly_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/021894_numcat_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/021894_numcat_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/021894_numcatpp_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/021894_numcatpp_spec.png"/>
            </td>
        </tr>
        <tr>
            <td>"NeutrnPluk"<br/>(harmonic)</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/077603_GT_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/077603_GT_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/077603_numonly_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/077603_numonly_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/077603_numcat_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/077603_numcat_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/077603_numcatpp_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/077603_numcatpp_spec.png"/>
            </td>
        </tr>
        <tr>
            <td>"Vio Solo 1"<br/>(harmonic)</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/003203_GT_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/003203_GT_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/003203_numonly_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/003203_numonly_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/003203_numcat_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/003203_numcat_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/003203_numcatpp_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/003203_numcatpp_spec.png"/>
            </td>
        </tr>
        <tr>
            <td>"Bass.Synth"<br/>(harmonic)</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080440_GT_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/080440_GT_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080440_numonly_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/080440_numonly_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080440_numcat_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/080440_numcat_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080440_numcatpp_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/080440_numcatpp_spec.png"/>
            </td>
        </tr>
        <tr>
            <td>"Japanany"<br/>(percussive)</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080737_GT_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/080737_GT_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080737_numonly_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/080737_numonly_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080737_numcat_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/080737_numcat_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080737_numcatpp_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/080737_numcatpp_spec.png"/>
            </td>
        </tr>
        <tr>
            <td>"CongaBongo"<br/>(percussive)</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/002094_GT_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/002094_GT_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/002094_numonly_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/002094_numonly_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/002094_numcat_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/002094_numcat_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/002094_numcatpp_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/002094_numcatpp_spec.png"/>
            </td>
        </tr>
        <tr> <!-- Use HQ .wav files for SFX -->
            <td>"R2.D2"<br/>(sfx)</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080285_GT_audio.wav" type="audio/wav" />
                </audio>
                <br />
                <img src="assets/synth_prog/080285_GT_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080285_numonly_audio.wav" type="audio/wav" />
                </audio>
                <br />
                <img src="assets/synth_prog/080285_numonly_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080285_numcat_audio.wav" type="audio/wav" />
                </audio>
                <br />
                <img src="assets/synth_prog/080285_numcat_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080285_numcatpp_audio.wav" type="audio/wav" />
                </audio>
                <br />
                <img src="assets/synth_prog/080285_numcatpp_spec.png"/>
            </td>
        </tr>
        <tr>
            <td>"LazerGun"<br/>(sfx)</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/074754_GT_audio.wav" type="audio/wav" />
                </audio>
                <br />
                <img src="assets/synth_prog/074754_GT_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/074754_numonly_audio.wav" type="audio/wav" />
                </audio>
                <br />
                <img src="assets/synth_prog/074754_numonly_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/074754_numcat_audio.wav" type="audio/wav" />
                </audio>
                <br />
                <img src="assets/synth_prog/074754_numcat_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/074754_numcatpp_audio.wav" type="audio/wav" />
                </audio>
                <br />
                <img src="assets/synth_prog/074754_numcatpp_spec.png"/>
            </td>
        </tr>
    </table>
</div>

<sup>*</sup> <small>Some inferred presets were out of tune hence their pitch has been manually adjusted to allow for fair comparisons.
The 'master tune' Dexed control, which was set to 50% during training, was used after preset inference.
This step could be automated using pitch estimation models such as CREPE[^2] on inferred audio samples.</small>

---
# Learning presets from multiple notes

Most synthesizers provide parameters which modulate sound depending on the intensity and pitch of played notes.
The paper introduces a convolutional structure for learning these parameters from multi-channel (multiple notes) spectrogram inputs.

The example below shows:
- an original preset from the test set
- a preset inferred by a model trained on single-note input data
- a preset inferred by a model trained on multiple-note input data, which has better learned how the sound should be modulated depending on note intensity

<div class="figure">
    <table>
        <tr>
            <th style="text-align: right">Note intensity: </th>
            <th style="text-align: center">20/127</th>
            <th style="text-align: center">64/127</th>
            <th style="text-align: center">127/127</th>
        </tr>
        <tr>
            <td>Original<br/>preset</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/dynamic_prog/midi_velocity/074483_GT_midi_060_020_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/dynamic_prog/midi_velocity/074483_GT_midi_060_020_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/dynamic_prog/midi_velocity/074483_GT_midi_060_064_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/dynamic_prog/midi_velocity/074483_GT_midi_060_064_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/dynamic_prog/midi_velocity/074483_GT_midi_060_127_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/dynamic_prog/midi_velocity/074483_GT_midi_060_127_spec.png"/>
            </td>
        </tr>
        <tr>
            <td>Single-channel<br/> input model</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/dynamic_prog/midi_velocity/074483_1midi_midi_060_020_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/dynamic_prog/midi_velocity/074483_1midi_midi_060_020_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/dynamic_prog/midi_velocity/074483_1midi_midi_060_064_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/dynamic_prog/midi_velocity/074483_1midi_midi_060_064_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/dynamic_prog/midi_velocity/074483_1midi_midi_060_127_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/dynamic_prog/midi_velocity/074483_1midi_midi_060_127_spec.png"/>
            </td>
        </tr>
        <tr>
            <td>Multi-channel<br/> input model</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/dynamic_prog/midi_velocity/074483_6stack_midi_060_020_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/dynamic_prog/midi_velocity/074483_6stack_midi_060_020_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/dynamic_prog/midi_velocity/074483_6stack_midi_060_064_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/dynamic_prog/midi_velocity/074483_6stack_midi_060_064_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/dynamic_prog/midi_velocity/074483_6stack_midi_060_127_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/dynamic_prog/midi_velocity/074483_6stack_midi_060_127_spec.png"/>
            </td>
        </tr>
    </table>
</div>

---

# Interpolation between presets

The VAE-based architecture is a *generative* model which can infer new presets.
Compared to previous works, our proposal is able to handle all synthesizer parameters simultaneously and allows to reduce the error on their latent encodings.

The following examples compare interpolations between two presets from the held-out test dataset:

- The "latent space" interpolation consists in encoding presets into the latent space using the RealNVP[^3] invertible regression flow in its inverse direction.
Then, a linear interpolation is performed on latent vectors, which can be converted back into presets using the regression flow in its forward direction.

- The "naive" interpolation consists in a linear interpolation between VST parameters.

### Interpolation example 1

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th>Start preset<br/>"NylonPick4"</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>End preset<br />"FmRhodes14"</th>
        </tr>
        <tr>
            <th></th>
            <td>Step 1/7</td>
            <td>Step 2/7</td>
            <td>Step 3/7</td>
            <td>Step 4/7</td>
            <td>Step 5/7</td>
            <td>Step 6/7</td>
            <td>Step 7/7</td>
        </tr>
        <tr> <!-- zK interp -->
            <th scope="row">
                Latent <br/>space
            </th>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/interpolation/055577to076380_zK_audio0.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/interpolation/055577to076380_zK_spec0.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/055577to076380_zK_audio1.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/055577to076380_zK_spec1.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/055577to076380_zK_audio2.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/055577to076380_zK_spec2.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/055577to076380_zK_audio3.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/055577to076380_zK_spec3.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/055577to076380_zK_audio4.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/055577to076380_zK_spec4.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/055577to076380_zK_audio5.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/055577to076380_zK_spec5.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/055577to076380_zK_audio6.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/055577to076380_zK_spec6.png"/>
            </td>
        </tr>
        <tr> <!-- naive interp -->
            <th scope="row">
                VST parameters<br/> (naive)
            </th>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/055577to076380_naive_audio0.mp3" type="audio/mp3" />
                </audio> <br/>
                <img src="assets/interpolation/055577to076380_naive_spec0.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/055577to076380_naive_audio1.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/055577to076380_naive_spec1.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/055577to076380_naive_audio2.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/055577to076380_naive_spec2.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/055577to076380_naive_audio3.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/055577to076380_naive_spec3.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/055577to076380_naive_audio4.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/055577to076380_naive_spec4.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/055577to076380_naive_audio5.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/055577to076380_naive_spec5.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/055577to076380_naive_audio6.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/055577to076380_naive_spec6.png"/>
            </td>
        </tr>
    </table>
</div>

### Interpolation example 2

<div class="figure">
    <table>
        <tr>
            <th></th>
            <th>Start preset<br/>"Dingle 1"</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>End preset<br />"Get it"</th>
        </tr>
        <tr>
            <th></th>
            <td>Step 1/7</td>
            <td>Step 2/7</td>
            <td>Step 3/7</td>
            <td>Step 4/7</td>
            <td>Step 5/7</td>
            <td>Step 6/7</td>
            <td>Step 7/7</td>
        </tr>
        <tr> <!-- zK interp -->
            <th scope="row">
                Latent <br/>space
            </th>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/interpolation/000348to004018_zK_audio0.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/interpolation/000348to004018_zK_spec0.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/000348to004018_zK_audio1.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/000348to004018_zK_spec1.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/000348to004018_zK_audio2.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/000348to004018_zK_spec2.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/000348to004018_zK_audio3.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/000348to004018_zK_spec3.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/000348to004018_zK_audio4.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/000348to004018_zK_spec4.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/000348to004018_zK_audio5.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/000348to004018_zK_spec5.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/000348to004018_zK_audio6.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/000348to004018_zK_spec6.png"/>
            </td>
        </tr>
        <tr> <!-- naive interp -->
            <th scope="row">
                VST parameters<br/> (naive)
            </th>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/000348to004018_naive_audio0.mp3" type="audio/mp3" />
                </audio> <br/>
                <img src="assets/interpolation/000348to004018_naive_spec0.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/000348to004018_naive_audio1.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/000348to004018_naive_spec1.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/000348to004018_naive_audio2.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/000348to004018_naive_spec2.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/000348to004018_naive_audio3.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/000348to004018_naive_spec3.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/000348to004018_naive_audio4.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/000348to004018_naive_spec4.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/000348to004018_naive_audio5.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/000348to004018_naive_spec5.png"/>
            </td>
            <td>
                <audio controls="" class=small_control> 
                    <source src="assets/interpolation/000348to004018_naive_audio6.mp3" type="audio/mp3" />
                </audio><br/>
                <img src="assets/interpolation/000348to004018_naive_spec6.png"/>
            </td>
        </tr>
    </table>
</div>


---
# Out-of-domain samples

The following examples show how the model can program the Dexed synthesizer using sounds from acoustic instruments or other synthesizers (with different synthesis abilities).

<div class="figure">
    <table>
        <tr>
            <th style="text-align: center">Original sound</th>
            <th style="text-align: center">Inferred preset</th>
        </tr>
        <tr>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/other_prog/reso_organish_60_85_GT_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/other_prog/reso_organish_60_85_GT_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/other_prog/reso_organish_60_85_inferred_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/other_prog/reso_organish_60_85_inferred_spec.png"/>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/other_prog/violin_60_85_GT_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/other_prog/violin_60_85_GT_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/other_prog/violin_60_85_inferred_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/other_prog/violin_60_85_inferred_spec.png"/>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/other_prog/chickenstalk_60_85_GT_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/other_prog/chickenstalk_60_85_GT_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/other_prog/chickenstalk_60_85_inferred_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/other_prog/chickenstalk_60_85_inferred_spec.png"/>
            </td>
        </tr>
        <tr>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/other_prog/roboclaver_60_85_GT_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/other_prog/roboclaver_60_85_GT_spec.png"/>
            </td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/other_prog/roboclaver_60_85_inferred_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/other_prog/roboclaver_60_85_inferred_spec.png"/>
            </td>
        </tr>
    </table>
</div>


---

[^2]: Jong Wook Kim, Justin Salamon, Peter Li, Juan Pablo Bello. "CREPE: A Convolutional Representation for Pitch Estimation", IEEE International Conference on Acoustics, Speech, and Signal Processing 2018. [https://arxiv.org/abs/1802.06182](https://arxiv.org/abs/1802.06182)
[^3]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio, “Density estimation using Real NVP,” International Conference on Learning Representations, 2017. [https://arxiv.org/abs/1605.08803](https://arxiv.org/abs/1605.08803)
