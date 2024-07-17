# Physics-informed Neural Networks
Physics-informed machine learning integrates seamlessly data and mathematical physics models.

### Installing
```
python -m venv .env
source .env/bin/activate
python -m pip install -r requirements.txt
```

## Harmonic Oscillator - 1 Dimension

Damped harmonic oscillator

<img src="src/03.Harmonic-oscillator/figures/oscillator.gif" width="350">

Differential equation
![\Large \frac{d^2x}{dt^2}+\frac{b}{m}\frac{dx}{dt}+\frac{k}{m}x=0](https://latex.codecogs.com/svg.latex?\Large&space;\dpi{150}\bg{white}\frac{d^2x}{dt^2}+\frac{b}{m}\frac{dx}{dt}+\frac{k}{m}x=0) 

Use a neural network
<img src="src/03.Harmonic-oscillator/figures/nn1D.gif" width="550"> <img src="src/03.Harmonic-oscillator/figures/NeuralNetword.png" width="250">


Use a physics-informed neural network
<img src="src/03.Harmonic-oscillator/figures/Physics-informedNeuralNetword.png" width="250">

<img src="src/03.Harmonic-oscillator/figures/pinn1D.gif" width="850">
