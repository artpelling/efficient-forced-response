from numpy import abs, argwhere, concatenate, log10, max, pad, round, sqrt, sum, zeros
from scipy.io import loadmat
from scipy.signal import resample
from pathlib import Path


def dB(vals):
    return 20 * log10(abs(vals))


def post_process(ir, fs, reduction=1, normalise=False, padding=False, crop=False):
    if crop:
        peak = max(abs(ir), axis=(0, 1))
        peak /= max(peak)
        idx = argwhere(peak > 1e-3)[:, 0]
        ir = ir[..., idx[0] : idx[-1]]
        idx = round([idx[0] / reduction, idx[-1] / reduction]).astype(int)
    else:
        idx = None

    if reduction:
        if reduction == 1:
            pass
        else:
            ir = resample(ir, ir.shape[2] // reduction, axis=-1)
            fs /= reduction

    if normalise:
        p, m, _ = ir.shape
        ir /= sqrt(sum(ir ** 2) / m / p)

    if padding:
        npad = ((0, 0), (0, 0), (0, ir.shape[-1]))
        ir = pad(ir, npad, constant_values=0)

    return ir, int(fs), idx


def load_MIRD_data(both=False):
    root = Path("data/MIRD")
    path = (
        root
        / "Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_3-3-3-8-3-3-3_1m_"
    )
    fs = 48000
    n = int(0.16 * fs)  # length of impulse responses
    pre = 630  # subtract predelay

    ir = zeros([8, 13, n])
    for i in range(13):
        angle = (270 + 15 * i) % 360
        mat = loadmat(str(path) + f"{angle:03d}" + ".mat")
        ir[:, i, :] = mat["impulse_response"].T[:, pre : pre + n]

    if both:
        path = (
            root
            / "Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_3-3-3-8-3-3-3_2m_"
        )
        ir = concatenate([ir, zeros([8, 13, n])], axis=1)
        for i in range(13):
            angle = (270 + 15 * i) % 360
            mat = loadmat(str(path) + f"{angle:03d}" + ".mat")
            ir[:, i + 13, :] = mat["impulse_response"].T[:, pre : pre + n]
        m = 26
    else:
        m = 13

    return ir, fs, m, 8
