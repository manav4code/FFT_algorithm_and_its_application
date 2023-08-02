import numpy as np
import matplotlib.pyplot as plt

def fft(x):
    N = len(x)
    getNextPow = int(pow(2,np.ceil(np.log2(N))))
    zeroPad = np.zeros(getNextPow - N)
    x = np.append(x,zeroPad)
    x = fft1(x)
    x = x[0:N]
    return x
    
def fft1(x):
        N = len(x)
        if N == 1:
            return x
        else:
            x_even = fft1(x[::2])
            x_odd = fft1(x[1::2])
            twiddle = np.exp((np.pi*(-2j)*(np.arange(N)))/N)

            x = np.concatenate([x_even + (twiddle[:int(N/2)])*x_odd, (x_even + (twiddle[int(N/2):]*x_odd))])
            return x

def ifft(x):
        f_con = np.conjugate(x);

        sig_x = np.conjugate(fft(f_con));
        sig_x = sig_x/x.shape[0]
        
        return sig_x
    
    
def psd(x):
    n = len(x)
    fhat = fft(x)
    PSD = fhat*np.conj(fhat) / n
    freq = np.arange(n)
    L = np.arange(1, np.floor(n/2), dtype='int')
    return PSD
    
def denoise(x):
    PSD = psd(x)
    fhat = fft(x)
    indices = np.where(PSD == np.max(PSD),1,0)
    PSDclean = PSD*indices
    fhat = indices*fhat
    ffilt = ifft(fhat)
    return x