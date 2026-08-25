#!/usr/bin/env python
"""Temporary Sabatier-to-Williams compatibility calibration.

Fit provisional Williams optical and current scales against the peak absolute
photocurrent of the legacy Sabatier model.  This script is deliberately separate
from production classes and never writes fitted values back into the model.

The legacy command is converted to surface irradiance using the production
0.39 mW/mm^2 convention and a unit spatial-profile factor.
Consequently, this is a compatibility calibration, not a biological parameter
estimate or a fit for a particular tissue location.
"""

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np

# Importing Mozaik can import matplotlib before pyplot is used below.  Select a
# writable cache and a headless backend first so the script also runs on clusters.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mozaik-calibration-matplotlib")
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares

from mozaik.sheets.direct_stimulator import (
    ChR2_H134R_system,
    ChrimsonR_system,
    LEGACY_SURFACE_IRRADIANCE_MW_PER_MM2,
    PROVISIONAL_WILLIAMS_EFFECTIVE_CURRENT_SCALER_PF,
    PROVISIONAL_WILLIAMS_T_OPTICAL,
    _mw_per_mm2_to_photons_per_s_per_cm2,
    _williams_current_density,
)

CALIBRATION_LABEL = "Temporary Sabatier→Williams compatibility calibration"
# Closed-loop legacy commands are documented on [0, 1].  Omit zero because the
# requested objective is logarithmic, and cover the positive operating range.
INPUT_AMPLITUDES = np.geomspace(0.01, 1.0, 8)
PULSE_DURATION_MS = 50.0
SAMPLING_PERIOD_MS = 1.0

SABATIER_DARK_STATE = np.array([0.0, 0.0, 0.2, 0.8, 0.0])
WILLIAMS_DARK_STATE = np.array([1.0, 0.0, 0.0, 0.0, 0.0])


def _sample_times():
    """Return the same 1 ms sample grid used for a 50-sample legacy command."""

    return np.arange(0.0, PULSE_DURATION_MS, SAMPLING_PERIOD_MS)


def sabatier_current_trace(input_amplitude):
    """Return the Sabatier current trace in nA for one legacy command level."""

    times = _sample_times()
    irradiance = np.full(
        times.size,
        LEGACY_SURFACE_IRRADIANCE_MW_PER_MM2 * input_amplitude,
        dtype=float,
    )
    photon_flux = _mw_per_mm2_to_photons_per_s_per_cm2(irradiance, 590.0)
    states = odeint(
        ChrimsonR_system,
        SABATIER_DARK_STATE,
        times,
        args=(photon_flux, SAMPLING_PERIOD_MS),
        hmax=SAMPLING_PERIOD_MS,
    )
    return 60.0 * (17.2 * states[:, 0] + 2.9 * states[:, 1]) / 2500.0


def williams_current_trace(input_amplitude, t_optical, effective_current_scaler_pf):
    """Return Williams current in nA for arbitrary positive candidate scales."""

    if t_optical <= 0.0 or effective_current_scaler_pf <= 0.0:
        raise ValueError("Williams calibration scales must be positive")

    times = _sample_times()
    irradiance = np.full(
        times.size,
        LEGACY_SURFACE_IRRADIANCE_MW_PER_MM2 * t_optical * input_amplitude,
        dtype=float,
    )
    states = odeint(
        ChR2_H134R_system,
        WILLIAMS_DARK_STATE,
        times,
        args=(irradiance, SAMPLING_PERIOD_MS),
        hmax=SAMPLING_PERIOD_MS,
    )
    return _williams_current_density(states) * effective_current_scaler_pf / 1000.0


def _peak_absolute(traces):
    return np.array([np.max(np.abs(trace)) for trace in traces])


def fit_compatibility_scales():
    """Fit 0 < T <= 1 and positive S by least squares in log-response space."""

    sabatier_traces = [
        sabatier_current_trace(amplitude) for amplitude in INPUT_AMPLITUDES
    ]
    sabatier_peaks = _peak_absolute(sabatier_traces)

    def residuals(log_scales):
        t_optical, current_scaler = np.exp(log_scales)
        williams_peaks = _peak_absolute(
            [
                williams_current_trace(amplitude, t_optical, current_scaler)
                for amplitude in INPUT_AMPLITUDES
            ]
        )
        return np.log(williams_peaks) - np.log(sabatier_peaks)

    initial_scales = np.array(
        [
            PROVISIONAL_WILLIAMS_T_OPTICAL,
            PROVISIONAL_WILLIAMS_EFFECTIVE_CURRENT_SCALER_PF,
        ]
    )
    result = least_squares(
        residuals,
        np.log(initial_scales),
        bounds=(np.log([1e-6, 1e-6]), np.log([1, 1e6])),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    if not result.success:
        raise RuntimeError(f"Compatibility fit failed: {result.message}")
    t_optical, current_scaler = np.exp(result.x)
    williams_traces = [
        williams_current_trace(amplitude, t_optical, current_scaler)
        for amplitude in INPUT_AMPLITUDES
    ]
    williams_peaks = _peak_absolute(williams_traces)
    log_errors = np.log(williams_peaks) - np.log(sabatier_peaks)
    objective = np.mean(log_errors**2)
    return (
        t_optical,
        current_scaler,
        objective,
        sabatier_traces,
        williams_traces,
        sabatier_peaks,
        williams_peaks,
        log_errors,
    )


def save_plots(
    output_dir,
    t_optical,
    current_scaler,
    sabatier_traces,
    williams_traces,
    sabatier_peaks,
    williams_peaks,
    log_errors,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    subtitle = f"T_optical={t_optical:.6g}, S={current_scaler:.6g} pF"

    dose_response_path = output_dir / "temporary_sabatier_to_williams_dose_response.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.loglog(
        INPUT_AMPLITUDES,
        sabatier_peaks,
        "o-",
        label="Sabatier",
    )
    ax.loglog(
        INPUT_AMPLITUDES,
        williams_peaks,
        "s--",
        label="Fitted Williams",
    )
    ax.set(
        xlabel="Legacy input amplitude",
        ylabel="Peak absolute photocurrent during pulse (nA)",
        title=f"{CALIBRATION_LABEL}\nDose response; {subtitle}",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(dose_response_path, dpi=180)
    plt.close(fig)

    error_path = output_dir / "temporary_sabatier_to_williams_ratio_error.png"
    fig, (ratio_ax, error_ax) = plt.subplots(
        2, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True
    )
    ratios = williams_peaks / sabatier_peaks
    ratio_ax.semilogx(INPUT_AMPLITUDES, ratios, "o-")
    ratio_ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.7)
    ratio_ax.set(
        ylabel="Williams / Sabatier peak",
        title=f"{CALIBRATION_LABEL}\nRatio and log error; {subtitle}",
    )
    ratio_ax.grid(True, which="both", alpha=0.3)
    error_ax.semilogx(INPUT_AMPLITUDES, log_errors, "o-")
    error_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    error_ax.set(
        xlabel="Legacy input amplitude",
        ylabel="log(Williams / Sabatier)",
    )
    error_ax.grid(True, which="both", alpha=0.3)
    fig.savefig(error_path, dpi=180)
    plt.close(fig)

    traces_path = output_dir / "temporary_sabatier_to_williams_traces.png"
    representative_indices = [0, len(INPUT_AMPLITUDES) // 2, -1]
    times = _sample_times()
    fig, axes = plt.subplots(
        len(representative_indices),
        1,
        figsize=(7.2, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    for ax, index in zip(axes, representative_indices):
        ax.plot(times, np.abs(sabatier_traces[index]), label="Sabatier")
        ax.plot(
            times,
            np.abs(williams_traces[index]),
            "--",
            label="Fitted Williams",
        )
        ax.set_ylabel("|current| (nA)")
        ax.set_title(f"Legacy input = {INPUT_AMPLITUDES[index]:.4g}")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Time from pulse onset (ms)")
    fig.suptitle(f"{CALIBRATION_LABEL}: 50 ms traces\n{subtitle}")
    fig.savefig(traces_path, dpi=180)
    plt.close(fig)

    return dose_response_path, error_path, traces_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sabatier_to_williams_calibration"),
        help="directory for calibration PNGs (default: %(default)s)",
    )
    args = parser.parse_args()

    print(f"=== {CALIBRATION_LABEL.upper()} ===")
    print("Compatibility fit only; these are not biological parameter estimates.")
    print(f"Legacy amplitudes: {INPUT_AMPLITUDES}")
    print(
        "Representative surface irradiance per unit command: "
        f"{LEGACY_SURFACE_IRRADIANCE_MW_PER_MM2:.6g} mW/mm^2"
    )

    (
        t_optical,
        current_scaler,
        objective,
        sabatier_traces,
        williams_traces,
        sabatier_peaks,
        williams_peaks,
        log_errors,
    ) = fit_compatibility_scales()
    paths = save_plots(
        args.output_dir,
        t_optical,
        current_scaler,
        sabatier_traces,
        williams_traces,
        sabatier_peaks,
        williams_peaks,
        log_errors,
    )

    print(f"Fitted T_optical = {t_optical:.12g}")
    print(f"Fitted effective_current_scaler_pF = {current_scaler:.12g}")
    print(f"Mean squared log-response objective = {objective:.12g}")
    print("No fitted values were written to production code.")
    for path in paths:
        print(f"Saved {path.resolve()}")


if __name__ == "__main__":
    main()
