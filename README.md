# Physics-informed Neural Networks
Physics-informed machine learning integrates seamlessly data and mathematical physics models.

### Installing
```
python -m venv .env
source .env/bin/activate
python -m pip install -r requirements.txt
```

## Harmonic Oscillator - 1 Dimension

#### Damped harmonic oscillator

<div align="center">
  <table>
    <tr>
      <td align="center" valign="middle">
        <img src="src/03.Harmonic-oscillator/figures/oscillator.gif" width="350">
      </td>
      <td align="center" valign="middle">
          <img src="https://latex.codecogs.com/svg.latex?\Large&space;\dpi{150}\bg{white}\frac{d^2x}{dt^2}+\frac{b}{m}\frac{dx}{dt}+\frac{k}{m}x=0" alt="\Large \frac{d^2x}{dt^2}+\frac{b}{m}\frac{dx}{dt}+\frac{k}{m}x=0">
      </td>
    </tr>
  </table>
</div>

#### Use a neural network

<div align="center">
  <table>
    <tr>
      <td align="center" valign="middle">
        <img src="src/03.Harmonic-oscillator/figures/NeuralNetword.png" width="350">
      </td>
      <td align="center" valign="middle">
        <img src="src/03.Harmonic-oscillator/figures/nn1D.gif" width="550">
      </td>
    </tr>
  </table>
</div>

#### Use a physics-informed neural network

<div align="center">
  <table>
    <tr>
      <td align="center" valign="middle">
        <img src="src/03.Harmonic-oscillator/figures/Physics-informedNeuralNetword.png" width="350">
      </td>
      <td align="center" valign="middle">
        <img src="src/03.Harmonic-oscillator/figures/pinn1D.gif" width="550">
      </td>
    </tr>
  </table>
</div>

<div align="left">
Original:[Ben Moseley](https://github.com/benmoseley)
</div>