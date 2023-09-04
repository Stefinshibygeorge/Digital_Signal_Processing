
from DSP_functions import ct_complex_exponential,sampling_signal_time_domain,dtft,dft


sig =  ct_complex_exponential([5900,5900*2,5900*0.5],
                             plot = 1,
                             values = 1,
                             start_time = 0,
                             stop_time = 0.001)  
  
print(f'Signal_values: \n{sig["signal_values"].round()}\n')
print(f'Magnitude_Spectrum: \n{sig["magnitude_spectrum"].round()}\n')
print(f'Phase_Spectrum: \n{sig["phase_spectrum"].round(1)}\n')
print(f'Real_Spectrum: \n{sig["real_spectrum"].round()}\n')
print(f'Imaginary_Spectrum: \n{sig["imaginary_spectrum"].round()}\n\n\n\n\n')   
    
        

sam_sig = sampling_signal_time_domain(sig = sig["signal_values"],
                          sampling_interval = 5,
                          sample_no = 16,
                          plot = 1,
                          values = 1)

print(f'Sampled_Signal_values: \n{sam_sig["sampled_signal_values"].round()}\n')
print(f'Magnitude_Spectrum: \n{sam_sig["magnitude_spectrum"].round()}\n')
print(f'Phase_Spectrum: \n{sam_sig["phase_spectrum"].round()}\n')
print(f'Real_Spectrum: \n{sam_sig["real_spectrum"].round()}\n')
print(f'Imaginary_Spectrum: \n{sam_sig["imaginary_spectrum"].round()}\n\n\n\n\n') 



dtft_sig = dtft(sam_sig["sampled_signal_values"],
            start_omg = -3.1415,
            stop_omg = 3.1415,
            plot = 1,
            values = 1)

print(f'DTFT_Signal_values: \n{dtft_sig["DTFT_of_signal"].round()}\n')
print(f'Magnitude_Spectrum: \n{dtft_sig["magnitude_spectrum"].round()}\n')
print(f'Phase_Spectrum: \n{dtft_sig["phase_spectrum"].round()}\n')
print(f'Real_Spectrum: \n{dtft_sig["real_spectrum"].round()}\n')
print(f'Imaginary_Spectrum: \n{dtft_sig["imaginary_spectrum"].round()}\n\n\n\n\n') 



dft_sig = dft(sam_sig["sampled_signal_values"],
          plot = 1,
          values  = 1)

print(f'DFT_Signal_values: \n{dft_sig["DFT_of_signal"].round()}\n')
print(f'Magnitude_Spectrums: \n{dft_sig["magnitude_spectrum"].round()}\n')
print(f'Phase_Spectrum: \n{dft_sig["phase_spectrum"].round()}\n')
print(f'Real_Spectrum: \n{dft_sig["real_spectrum"].round()}\n')
print(f'Imaginary_Spectrum: \n{dft_sig["imaginary_spectrum"].round()}\n\n\n\n\n') 
   