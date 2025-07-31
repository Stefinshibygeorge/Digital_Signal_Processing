# Signal Processing Functions

This repository provides a collection of functions for generating, sampling, and analyzing complex exponential signals in both time and frequency domains. It includes implementations of DTFT, DFT, and FFT, as well as signal visualization tools.

---

## 1. `ct_complex_exponential()`

Generates a continuous-time complex exponential signal (simulated at high resolution).

### Parameters:

* **freq** (list): List of frequencies in Hz (e.g., `[3400]`, `[3400, 2300]`).
* **start\_time** (float): Start time of the signal.
* **stop\_time** (float): End time of the signal.
* **plot** (bool): Whether to plot magnitude, phase, real, and imaginary components.
* **values** (bool): Whether to return signal data as a dictionary.

### Returns (if `values=True`):

* Dictionary with:

  * `signal_values`
  * `magnitude_spectrum`
  * `phase_spectrum`
  * `real_spectrum`
  * `imaginary_spectrum`

---

## 2. `sampling_signal_time_domain()`

Samples a time-domain signal and plots the sample spectra.

### Parameters:

* **sig** (ndarray): The input signal to be sampled.
* **sampling\_interval** (int): The gap between samples.
* **sample\_no** (int): Number of samples to consider.
* **plot** (bool): Whether to plot sampled signal.
* **values** (bool): Whether to return sampled data.

### Returns (if `values=True`):

* Dictionary with:

  * `sampled_signal_values`
  * `magnitude_spectrum`
  * `phase_spectrum`
  * `real_spectrum`
  * `imaginary_spectrum`

---

## 3. `dtft()`

Computes the Discrete-Time Fourier Transform (DTFT) of a sampled signal.

### Parameters:

* **sam\_sig** (ndarray): Sampled signal values.
* **start\_omg** (float): Starting frequency in radians/sample.
* **stop\_omg** (float): Stopping frequency.
* **plot** (bool): Whether to show DTFT plots.
* **values** (bool): Whether to return DTFT data.

### Returns (if `values=True`):

* Dictionary with:

  * `DTFT_of_signal`
  * `magnitude_spectrum`
  * `phase_spectrum`
  * `real_spectrum`
  * `imaginary_spectrum`

---

## 4. `dft()`

Computes the Discrete Fourier Transform (DFT) using SciPy FFT.

### Parameters:

* **sam\_sig** (ndarray): Sampled signal values.
* **plot** (bool): Whether to plot the DFT.
* **values** (bool): Whether to return DFT data.

### Returns (if `values=True`):

* Dictionary with:

  * `DFT_of_signal`
  * `magnitude_spectrum`
  * `phase_spectrum`
  * `real_spectrum`
  * `imaginary_spectrum`

---

## 5. `fft()`

Implements Radix-2 FFT algorithm for N-point DFT.

### Parameters:

* **signal** (list): Input discrete signal.
* **N** (int): Number of points for DFT.
* **plot** (bool): Whether to show FFT plots.
* **values** (bool): Whether to return FFT data.

### Returns (if `values=True`):

* Dictionary with:

  * `fft_values`
  * `magnitude_spectrum`
  * `phase_spectrum`
  * `real_spectrum`
  * `imaginary_spectrum`

---

## Notes

* Signals are generated and sampled using NumPy.
* All spectrum plots are handled via `matplotlib.pyplot`.
* Sampling assumes 100,000 points/sec for approximating continuous-time behavior.

---

## Dependencies

* NumPy
* Matplotlib
* SciPy (for `fft.fft` in DFT)

Install with:

```bash
pip install numpy matplotlib scipy
```

---

Feel free to modify or extend any function to suit your applications in signal processing or digital communication experiments.
