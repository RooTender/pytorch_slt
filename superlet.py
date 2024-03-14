# Time-frequency analysis with superlets
# Based on 'Time-frequency super-resolution with superlets'
# by Moca et al., 2021 Nature Communications
#
# Implementation by Harald Bârzan and Richard Eugen Ardelean
# Optimized for PyTorch by Hubert Lewandowski

#
# Note: for runs on multiple batches of data, the class SuperletTransform can be instantiated just once
# this saves time and memory allocation for the wavelets and buffers
#


import torch
from fft import FFTConvolve


# spread, in units of standard deviation, of the Gaussian window of the Morlet wavelet
MORLET_SD_SPREAD = 6

# the length, in units of standard deviation, of the actual support window of the Morlet
MORLET_SD_FACTOR = 2.5




def computeWaveletSize(center_freq: torch.Tensor, cycles: int, sampling_rate: int):
    """
    Compute the size in samples of a morlet wavelet.
    Arguments:
        center_freq - center frequency in Hz
        cycles - number of cycles
        sampling_rate - sampling rate in Hz
    """
    sd = (cycles / 2) * (1 / torch.abs(center_freq)) / MORLET_SD_FACTOR
    return int(2 * torch.floor(torch.round(sd * sampling_rate * MORLET_SD_SPREAD) / 2) + 1)


def gausswin(size: int, alpha: int):
    """
    Create a Gaussian window.
    Arguments:
        size - size of the window
        alpha - controls gaussian curve
    """
    halfSize    = size // 2
    idiv        = alpha / halfSize

    t = (torch.arange(size, dtype=torch.float64) - halfSize) * idiv
    return torch.exp(-(t * t) * 0.5)


def morlet(center_freq: int, cycles: int, sampling_rate: int):
    """
    Create an analytic Morlet wavelet.
    Arguments:
        center_freq - center frequency in Hz
        cycles - number of cycles
        sampling_rate - sampling rate in Hz
    """
    size    = computeWaveletSize(center_freq, cycles, sampling_rate)
    half    = size // 2
    gauss   = gausswin(size, MORLET_SD_SPREAD / 2)
    igsum   = 1 / gauss.sum()
    ifs     = 1 / sampling_rate

    t = (torch.arange(size, dtype=torch.float64) - half) * ifs
    return gauss * torch.exp(2 * torch.pi * center_freq * t * 1j) * igsum

def fractional(x):
    """
    Get the fractional part of the scalar value x.
    """
    return x - int(x)


class SuperletTransform:
    """
    Class used to compute the Superlet Transform of input data.
    """

    def __init__(   self,
                    inputSize,
                    samplingRate,
                    frequencyRange,
                    frequencyBins,
                    baseCycles,
                    superletOrders,
                    frequencies = None):
        """
        Initialize the superlet transform. 
        Arguments:
            inputSize: size of the input in samples
            samplingRate: the sampling rate of the input signal in Hz
            frequencyRange: tuplet of ascending frequency points, in Hz
            frequencyBins: number of frequency bins to sample in the interval frequencyRange
            baseCycles: number of cycles of the smallest wavelet (c1 in the paper)
            superletOrders: a tuple containing the range of superlet orders, linearly distributed along frequencyRange
            frequencies: specific list of frequencies - can be provided in stead of frequencyRange (it is ignored in this case)
        """
        # clear to reinit
        self.clear()

        # initialize containers
        if frequencies is not None:
            frequencyBins = len(frequencies)
            self.frequencies = torch.tensor(frequencies, dtype=torch.float64)
        else:
            self.frequencies = torch.linspace(start=frequencyRange[0], end=frequencyRange[1], steps=frequencyBins)

        self.inputSize      = inputSize
        self.orders         = torch.linspace(start=superletOrders[0], end=superletOrders[1], steps=frequencyBins)
        self.convBuffer     = torch.zeros(inputSize, dtype=torch.complex128)
        self.poolBuffer     = torch.zeros(inputSize, dtype=torch.float64)
        self.superlets      = []

        # create wavelets
        for iFreq in range(frequencyBins):
            centerFreq  = self.frequencies[iFreq]
            nWavelets   = int(torch.ceil(self.orders[iFreq]).item())

            self.superlets.append([])
            for iWave in range(nWavelets):

                # create morlet wavelet
                self.superlets[iFreq].append(morlet(centerFreq, (iWave + 1) * baseCycles, samplingRate))


    def __del__(self):
        """
        Destructor.
        """
        self.clear()


    def clear(self):
        """
        Clear the transform.
        """
        # fields
        self.inputSize   = None
        self.superlets   = None
        self.poolBuffer  = None
        self.convBuffer  = None
        self.frequencies = None
        self.orders      = None


    
    def transform(self, data: torch.Tensor):
        """
        Apply the transform to a buffer or list of buffers.
        Arguments:
            data - a PyTorch tensor of input data
        """

        # compute number of arrays to transform
        if data.dim == 1:
            if data.size(0) != self.inputSize:
                raise "Input data must meet the defined input size for this transform."
            
            result = torch.zeros((self.inputSize, len(self.frequencies)), dtype=torch.float64)
            self.transformOne(data, result)
            return result

        else:
            n       = torch.prod(torch.tensor(data.shape[:-1])).item()
            insize  = int(data.shape[len(data.shape) - 1])

            if insize != self.inputSize:
                raise "Input data must meet the defined input size for this transform."
            
            # reshape to data list
            datalist = data.reshape(-1, insize)
            result = torch.zeros(len(self.frequencies), self.inputSize, dtype=torch.float64)

            for i in range(n):
                self.transformOne(datalist[i, :], result)

            return result / n


    def transformOne(self, data: torch.Tensor, accumulator: torch.Tensor):
        """
        Apply the superlet transform on a single data buffer.
        Arguments:
            data: A 1xInputSize array containing the signal to be transformed.
            accumulator: a spectrum to accumulate the resulting superlet transform
        """
        accumulator.reshape((len(self.frequencies), self.inputSize))
        fft = FFTConvolve(mode="same")

        for iFreq in range(len(self.frequencies)):
            
            # init pooling buffer
            self.poolBuffer = torch.ones_like(self.poolBuffer)

            if len(self.superlets[iFreq]) > 1:
                
                # superlet
                nWavelets   = int(torch.floor(self.orders[iFreq]))
                rfactor     = 1.0 / nWavelets

                for iWave in range(nWavelets):
                    self.convBuffer = fft(data, self.superlets[iFreq][iWave])
                    self.poolBuffer *= 2 * torch.abs(self.convBuffer) ** 2

                if fractional(self.orders[iFreq]) != 0 and len(self.superlets[iFreq]) == nWavelets + 1:

                    # apply the fractional wavelet
                    exponent = self.orders[iFreq] - nWavelets
                    rfactor = 1 / (nWavelets + exponent)

                    self.convBuffer = fft(data, self.superlets[iFreq][nWavelets])
                    self.poolBuffer *= (2 * torch.abs(self.convBuffer) ** 2) ** exponent

                # perform geometric mean
                accumulator[iFreq, :] += self.poolBuffer ** rfactor

            else:
                # wavelet transform
                accumulator[iFreq, :] += (2 * torch.abs(fft(data, self.superlets[iFreq][0])) ** 2).to(torch.float64)


# main superlet function
def superlets(data: torch.Tensor,
              sampling_rate: int,
              freqs: torch.Tensor,
              base_cycles: int,
              orders: int | tuple[int, int]):
    """
    Perform fractional adaptive superlet transform (FASLT) on a list of trials. 
    Arguments:
        data: a PyTorch tensor of data. The rightmost dimension of the data is the trial size. The result will be the average over all the spectra.
        sampling_rate: the sampling rate in Hz
        freqs: list of frequencies of interest
        base_cycles: base number of cycles parameter
        orders: the order (for SLT) or order range (for FASLT), spanned across the frequencies of interest
    Returns: a tensor containing the average superlet spectrum
    """
    # determine buffer size
    bufferSize = data.size(-1)

    # make order parameter
    if isinstance(orders, int):
        orders = (orders, orders)

    # build the superlet analyzer
    faslt = SuperletTransform(  inputSize        = bufferSize, 
                                frequencyRange   = None, 
                                frequencyBins    = None, 
                                samplingRate     = sampling_rate, 
                                frequencies      = freqs, 
                                baseCycles       = base_cycles, 
                                superletOrders   = orders)
        
    # apply transform
    result = faslt.transform(data)
    faslt.clear()

    return result


def amplitude_to_db(S, ref_power=None, amin=1e-5, top_db=80.0):
    """
    Convert an amplitude spectrogram to dB-scaled spectrogram in PyTorch.
    
    Args:
        S (Tensor): input amplitude spectrogram.
        ref_power (float or callable or None): reference power for dB scale. If callable, ref_power(S) is used.
        amin (float): minimum threshold for S.
        top_db (float): threshold the output at top_db below the peak.

    Returns:
        Tensor: Spectrogram in dB.
    """
    S_clamped = torch.clamp(S, min=amin)
    
    if callable(ref_power):
        ref_value = ref_power(S_clamped)
    elif ref_power is None:
        ref_value = S_clamped.max()
    else:
        ref_value = max(amin, ref_power)

    log_spec = 10.0 * torch.log10(S_clamped)
    ref_log_spec = 10.0 * torch.log10(torch.tensor(ref_value, device=S.device))

    S_db = log_spec - ref_log_spec

    if top_db is not None:
        S_db = torch.clamp(S_db, min=S_db.max() - top_db)

    return S_db
