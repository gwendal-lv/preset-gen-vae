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

# Interpolation

The following examples compare interpolations between two presets.

- The "latent space" interpolation consists in encoding presets into the latent space using the RealNVP invertible regression flow in its inverse direction.
Then, a linear interpolation is performed on latent vectors z, which can be converted back into presets using the regression flow in its forward direction.

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