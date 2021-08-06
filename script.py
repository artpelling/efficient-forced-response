from sid import BlockHankelMatrix, era, impulse_response, rsvd
from utils import dB, load_MIRD_data, norm, post_process
from numpy import array, concatenate, set_printoptions, sqrt, where, zeros
from numpy.fft import fft, fftfreq, rfft, rfftfreq
import matplotlib as mpl
import matplotlib.pyplot as plt

set_printoptions(precision=2)


# PARAMETERS #
r = 2000  # order of low-rank approximation with RSVD
orders = [
    2000,
    1000,
    500,
]  # orders of the reduced realizations that are constructed
flim = None  # window for frequency-limited realization. use numpy array e.g. array([100,1000])
plots = True
IOs = (
    (6, 3),
    (10, 6),
    (16, 1),
)  # input-output transmission channel indices to be plotted

if __name__ == "__main__":
    data, fs, m, p = load_MIRD_data(both=True)
    h, fs, idx = post_process(
        data,
        fs,
        reduction=2,
        normalise=True,
    )
    n = h.shape[-1]
    H = BlockHankelMatrix(h)
    U, S, Vt = rsvd(H, r, n_oversamples=int(r / 10), n_subspace_iters=2)

    realizations = dict.fromkeys(orders)
    for order in orders:
        A, B, C = era(U, S, Vt, p, m, dt=1 / fs, r=order, flim=flim)
        realizations[order] = (A, B, C)

    if plots:
        mpl.rcParams["lines.linewidth"] = 1.2
        mpl.rcParams.update({"font.size": 14})
        mpl.rcParams["mathtext.fontset"] = "stix"
        mpl.rcParams["font.family"] = "STIXGeneral"
        mpl.rc("text", usetex=True)

        def format_ax(ax, last=False, text=None):
            ocs = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
            ax.grid(True)
            ax.set_xlim([63, 12000])
            ax.set_xticks(ocs)
            ax.set_xticklabels([str(oc) for oc in ocs])
            # ax.set_xticklabels(["" for _ in ocs])
            ax.minorticks_off()
            ax.set_ylim([-21, 9])
            ax.set_yticks([-18, -12, -6, 0, 6])
            ax.set_ylabel(r"Magnitude [dB]")
            if text is not None:
                ax.text(80, 2.5, text, fontsize="16")
            if last:
                ax.set_xlabel(r"Frequency [Hz]")
                ax.legend(loc=4, framealpha=1)

        num_plots = len(IOs)
        f = rfftfreq(3840, 1 / fs)
        c = dict(
            zip(
                orders,
                ["tab:" + x for x in ["red", "blue", "green", "orange"][: len(orders)]],
            )
        )
        fig, axes = plt.subplots(num_plots, 1, figsize=(8, 12), dpi=300)
        k = 0

        text = [r"(\textbf{a})", r"(\textbf{b})", r"(\textbf{c})"]

        imps = dict.fromkeys(orders)
        for order in orders:
            imps[order] = impulse_response(*realizations[order], n)
            if flim is not None:
                ff = fftfreq(n, 1 / fs)
                idx = where((abs(ff) >= flim[0]) & (abs(ff) <= flim[1]))
                err = norm(fft(h)[..., idx] - fft(imps[order])[..., idx]) / sqrt(
                    len(idx[0])
                )
            else:
                err = norm(h - imps[order])
            print(f"h2err: {err}")

        for ax in axes:
            i, j = IOs[k]
            ax.semilogx(
                f,
                dB(rfft(h[j, i])),
                linestyle="-.",
                color="tab:gray",
                label=r"reference",
            )

            for order in orders:
                ax.semilogx(
                    f,
                    dB(rfft(imps[order][j, i])),
                    linestyle="-",
                    color=c[order],
                    label=fr"$n={order}$",
                )

            t = text[k]
            k += 1

            if k == num_plots:
                format_ax(ax, last=True, text=t)
            else:
                format_ax(ax, text=t)

        plt.subplots_adjust(hspace=0.1, top=0.98, bottom=0.05, right=0.97, left=0.09)
        plt.show()
