---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: splash
classes: wide

---

<link rel="stylesheet" href="assets/css/styles.css">

DAFx21 paper (under review) accompanying material

TEMP WEBSITE

---

# Automatic synthesizer programming

These first examples demonstrate how our model can be used to program a [Dexed FM synthesizer](https://github.com/asb2m10/dexed) (Yamaha DX7 clone) to reproduce a given audio input.
Input sounds were generated using original presets from the held-out test set of our [30k presets dataset](https://github.com/gwendal-lv/preset-gen-vae/blob/main/synth/dexed_presets.sqlite).
Each preset contains 144 learnable parameters.


<!---
Our architecture is based on a spectral convolutional VAE and integrates an additional decoder to perform a regression on synthesizer parameters.
The regression neural network is based on RealNVP [^3] invertible flows.
The general architecture is inspired by Esling et al.[^1], 2020. Modifications and improvements over this original proposal are detailed in our paper.
-->

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
            <td>-</td>
            <td>-</td>
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
            <td>-</td>
            <td>-</td>
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
            <td>-</td>
            <td>-</td>
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
            <td>-</td>
            <td>-</td>
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
            <td>-</td>
            <td>-</td>
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
            <td>-</td>
            <td>-</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/002094_numcatpp_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/002094_numcatpp_spec.png"/>
            </td>
        </tr>
        <!-- 
        <tr>
            <td>"DigiPerc."<br/>(percussive)</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080614_GT_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/080614_GT_spec.png"/>
            </td>
            <td>-</td>
            <td>-</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080614_numcatpp_audio.mp3" type="audio/mp3" />
                </audio>
                <br />
                <img src="assets/synth_prog/080614_numcatpp_spec.png"/>
            </td>
        </tr>  -->
        <tr> <!-- Use HQ .wav files for SFX -->
            <td>"R2.D2"<br/>(sfx)</td>
            <td>
                <audio controls class=small_control> 
                    <source src="assets/synth_prog/080285_GT_audio.wav" type="audio/wav" />
                </audio>
                <br />
                <img src="assets/synth_prog/080285_GT_spec.png"/>
            </td>
            <td>-</td>
            <td>-</td>
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
            <td>-</td>
            <td>-</td>
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

<sup>*</sup> <small>Some inferred presets were slightly out of tune, hence their pitch has been manually adjusted to allow for fair comparisons.
The 'master tune' Dexed control, which was set to 50% during training, was used after preset inference.
This step could be automated using pitch estimation models such as CREPE[^2] on inferred audio samples.</small>

---
# Dynamic parameters

TODO compare our best single-channel spectrogram model to our multi-channel proposal.

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
# Preset inference from natural samples

TODO with different pitches and velocities

- guitar 
- oboe/clarinet

- moog synth?


---

[^1]: Philippe Esling, Naotake Masuda, Adrien Bardet, Romeo Despres, and Axel Chemla-Romeu-Santos, “Flow synthesizer: Universal audio synthesizer control with normalizing flows,” Applied Sciences, vol. 10, no. 1, pp. 302, 2020. Originally published as a DAFx19 conference paper: [https://arxiv.org/abs/1907.00971](https://arxiv.org/abs/1907.00971)
[^2]: Jong Wook Kim, Justin Salamon, Peter Li, Juan Pablo Bello. "CREPE: A Convolutional Representation for Pitch Estimation", ICASSP 2018. [https://arxiv.org/abs/1802.06182](https://arxiv.org/abs/1802.06182)
[^3]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio, “Density estimation using Real NVP,” International Conference on Learning Representations, 2017. [https://arxiv.org/abs/1605.08803](https://arxiv.org/abs/1605.08803)
