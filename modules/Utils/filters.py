import numpy as np
import pandas as pd
import pywt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import savgol_filter
import statsmodels as sm


def lowpassfilter(signal, thresh = 0.63, wavelet="db5")-> np.array:
    """Perform a wavelet low pass filter on the data.

    Args:
        signal (numpy.array): The data to denoise.
        thresh (float, optional): The wavelet threshold to denoise data, higher threshold -> more denoise. Defaults to 0.63.
        wavelet (str, optional): Wavelet type 'sym5', 'coif5', 'bior2.4': . Defaults to "db5".

    Returns:
        numpy.array: Denoised data
    """
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    if len(signal) % 2==0:
        return reconstructed_signal
    else:
        return reconstructed_signal[1:]
    
    
    
def fft_denoiser(x, n_components=0.0005, to_real=True)-> np.array:
    """Fast fourier transform denoiser.
    
    Denoises data using the fast fourier transform.
    
    Parameters
    ----------
    x : numpy.array
        The data to denoise.
    n_components : int
        The value above which the coefficients will be kept.
    to_real : bool, optional, default: True
        Whether to remove the complex part (True) or not (False)
        
    Returns
    -------
    clean_data : numpy.array
        The denoised data.    
    """
    n = len(x)
    
    # compute the fft
    fft = np.fft.fft(x, n)
    
    # compute power spectrum density
    # squared magnitud of each fft coefficient
    PSD = fft * np.conj(fft) / n
    
    # keep high frequencies
    _mask = PSD > n_components
    fft = _mask * fft
    
    # inverse fourier transform
    clean_data = np.fft.ifft(fft)
    
    if to_real:
        clean_data = clean_data.real
    
    return pd.Series(clean_data)


def polynomialRegression(data, order,auto_order=True)->np.array:
    """Perform a polynomial regression on an input

    Args:
        data (numpy.array): The signal to denoise.
        order (int): The polynomial degree.
        auto_order (bool, optional): Whether we want to find the best fitting order or if we want to apply regression the a specific degree. Defaults to True.

    Returns:
        np.array: The denoised signal
    """
    indexs_2 = np.array([i for i in range(len(data))])
    indexs_2 = np.reshape(indexs_2, newshape=(1,-1))[0]
    if auto_order==True:
        RMSEs = []
        for i in range(order):
            features = PolynomialFeatures(degree=i+1)
            x_train_transformed = features.fit_transform(indexs_2.reshape(-1, 1))
            
            model = LinearRegression().fit(x_train_transformed, data)

            train_pred = model.predict(x_train_transformed)
            rmse_poly_4_train = mean_squared_error(data, train_pred, squared = False)
            RMSEs.append(rmse_poly_4_train)

        features = PolynomialFeatures(degree=RMSEs.index(min(RMSEs))+1)
        
    else:
        features = PolynomialFeatures(order)

    x_train_transformed = features.fit_transform(indexs_2.reshape(-1, 1))
    model = LinearRegression().fit(x_train_transformed, data) 
    x_test_transformed = features.fit_transform(indexs_2.reshape(-1, 1))

    train_pred = model.predict(x_train_transformed)
    test_pred = model.predict(x_test_transformed)

    return test_pred


def filterData(data, method='dwt')->np.array:
    """function the filter a signal with a specific method.

    Args:
        data (np.array): The input signal to denoise.
        method (str): The filtering/denoising method. Default to 'dwt'.

    Returns:
        np.array: The filtered signal.
    """
    if method=='savgol':
        return savgol_filter(data, 19, 2,mode='nearest',deriv=0)
    elif method=='fft':
        return  fft_denoiser(data, 20)
    elif method=='poly':
        return  polynomialRegression(data, data,order=30)
    elif method=='dwt':
        return  lowpassfilter(data,thresh=0.1)
    elif method=='hpf':
        cycle, trend = sm.tsa.filters.hpfilter(data)
        return trend 
    

def explicit_heat_smooth(prices: np.array,
                         t_end: float = 3.0) -> np.array:
    '''
    Smoothen out a time series using a simple explicit finite difference method.
    The scheme uses a first-order method in time, and a second-order centred
    difference approximation in space. The scheme is only numerically stable
    if the time-step 0<=k<=1.
    
    The prices are fixed at the end-points, so the interior is smoothed.

    Parameters
    ----------
    prices : np.array
        The price to smoothen
    t_end : float
        The time at which to terminate the smootheing (i.e. t = 2)
        

    Returns
    -------
    P : np.array
        The smoothened time-series

    '''
    
    k = 0.1 # Time spacing
    
    # Set up the initial condition
    P = prices
    
    t = 0
    while t < t_end:
        # Solve the finite difference scheme for the next time-step
        P = k*(P[2:] + P[:-2]) + P[1:-1]*(1-2*k)
        
        # Add the fixed boundary conditions since the above solves the interior
        # points only
        P = np.hstack((
            np.array([prices[0]]),
            P,
            np.array([prices[-1]]),
        ))
        t += k

    return P


def heat_analytical_smooth(prices: np.array, 
                           t: float = 3.0, 
                           m: int = 200) -> np.array:
    '''
    Find the analytical solution to the heat equation
    
    See: https://tutorial.math.lamar.edu/classes/de/heateqnnonzero.aspx

    Parameters
    ----------
    prices : np.array
        The price to smoothen.
    t : float
        The time at which to terminate the smootheing (i.e. t = 2)
    m : int
        The amount of terms in the solution's Fourier series

    Returns
    -------
    np.array
        The analytical solution to the heat equation

    '''
    
    p0 = prices[0]
    pn = prices[-1]
    
    n = prices.shape[0]
    x = np.arange(0, n, dtype = np.float32)
    M = np.arange(1, m, dtype = np.float32)
    
    L = n-1
    u_e = p0 + (pn - p0)*x/L
    
    mx = M.reshape(-1, 1)@x.reshape(1, -1)
    sin_m_pi_x = np.sin(mx*np.pi/L)
    
    # Calculate the B_m terms using numerical quadrature (trapezium rule)
    bm = 2*np.sum(
        (sin_m_pi_x*(prices-u_e)).T,
        axis = 0,
    )/n

    return u_e + np.sum(
        (bm*np.exp(-t*(M*np.pi/L)**2)).reshape(-1, 1)*sin_m_pi_x,
        axis = 0,
    )
