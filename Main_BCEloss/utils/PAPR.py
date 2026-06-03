import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from legends import CCDF_label_name_dict, CCDF_marker_dict
import re

def emprical_papr(x, T, epsilon_P):
    
    #
    # last_dim = x.shape[1]
    # new_last_dim = (last_dim // T) * T     
    # x = x[:, :new_last_dim]
    # x_reshaped = tf.reshape(x, [x.shape[0],-1,T])

    # Compute instantaneous power
    p_t = tf.abs(x)**2
    
    # compute mean power
    p_bar = tf.expand_dims(tf.reduce_mean(tf.abs(x)**2,  axis=[-1]), axis=-1) 

    # Compute max
    p_t_max = tf.math.maximum((p_t / p_bar) - tf.expand_dims(tf.pow(10.0, epsilon_P / 10.0),axis=-1), tf.constant(0.0,dtype=tf.float32))

    return tf.reduce_mean(p_t_max,axis=-1)

def emprical_ccdf_plotter(
    x_tf,
    rms_ds=None,                 # shape (B,) or None
    n_select=200,                  # how many traces to pick if rms_ds is provided
    to_db=True,
    x_range=(0.0, 10.0),         # fixed x-axis range (in dB if to_db=True)
    num_points=2000,             # all curves returned with this length
    papr_ccdf_levels=None        # optional markers from each curve, e.g. (1e-3,)
):
    """
    Empirical POWER CCDF per selected trace, evaluated on a FIXED x_plot grid.
    - No union-support interpolation.
    - Tails with no samples produce CCDF=0 on that grid.
    - All returned curves have identical length = num_points.

    Returns dict:
      {
        "x_plot": (num_points,),
        "ccdf_curves": {label: (num_points,)},
        "selected_indices": [...],
        "papr_markers": {label: {level: x_at_level}}  # optional
      }
    """
    data = x_tf.numpy()
    B, N = data.shape

    # choose indices
    if rms_ds is None:
        selected_indices = [0]
        selected_labels = ["b0"]
    else:
        rms_ds = np.asarray(rms_ds).reshape(-1)
        valid = np.isfinite(rms_ds) & (rms_ds > 0)
        valid_idx = np.where(valid)[0]
        order = np.argsort(rms_ds[valid_idx])[::-1]  # high -> low
        k = max(1, int(n_select))
        L = order.size

        # indices evenly spaced from highest to lowest in the sorted order
        pick = np.linspace(0, L - 1, k).astype(int)
        chosen = valid_idx[order[pick]]

        selected_indices = chosen.tolist()
        selected_labels = [f"b{idx} (rms_ds={rms_ds[idx]:.3g})" for idx in selected_indices] # include the batch index
        # selected_labels = [f"DS [rms]={rms_ds[idx]:.3g}" for idx in selected_indices]

    # fixed x grid for ALL curves
    x_min, x_max = x_range


    x_plot = np.linspace(x_min, x_max, int(num_points))

    ccdf_curves = {}
    papr_markers = {}

    for label, b in zip(selected_labels, selected_indices):
        x = data[b]

        # instantaneous power
        p = np.abs(x) ** 2
        p = p[np.isfinite(p)]
        if p.size == 0:
            continue

        # per-trace normalization (mean power)
        avg = np.mean(p)
        if not np.isfinite(avg) or avg <= 0:
            continue

        ratio = p / avg
        ratio = ratio[np.isfinite(ratio)]

        if to_db:
            # remove non-positive -> would become -inf in dB
            ratio = ratio[ratio > 0]
            if ratio.size == 0:
                continue
            ratio = np.maximum(ratio.astype(np.float64), np.finfo(np.float64).tiny)
            s = 10.0 * np.log10(ratio)
            s = s[np.isfinite(s)]
        else:
            s = ratio

        if s.size == 0:
            continue

        # sort once; compute CCDF on fixed grid via searchsorted (NO interpolation)
        s = np.sort(s)
        n = s.size

        # idx = number of samples <= x (side='right'); samples > x are n-idx
        idx = np.searchsorted(s, x_plot, side='right')
        ccdf = (n - idx) / n  # in [0,1], equals 0 beyond max(s)

        ccdf_curves[label] = ccdf

        if papr_ccdf_levels is not None:
            papr_markers[label] = _papr_x_at_ccdf_levels(x_plot, ccdf, papr_ccdf_levels)

    out = {
        "x_plot": x_plot,
        "ccdf_curves": ccdf_curves,
        "selected_indices": selected_indices,
        "to_db": to_db,
        "x_range": x_range,
    }
    if papr_ccdf_levels is not None:
        out["papr_markers"] = papr_markers
        out["papr_ccdf_levels"] = tuple(papr_ccdf_levels)

    return out

def _papr_x_at_ccdf_levels(x_plot, ccdf_curve, levels):
    """
    Return x such that CCDF(x)=level.
    If the curve cannot reach that level on the provided grid, returns NaN.
    """
    x = np.asarray(x_plot)
    y = np.asarray(ccdf_curve)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return {lvl: np.nan for lvl in levels}

    # CCDF decreases with x; reverse so y increases for np.interp
    y_rev = y[::-1]
    x_rev = x[::-1]

    out = {}
    for lvl in levels:
        # if lvl is outside [min(y), max(y)] cannot interpolate
        if lvl > y.max() or lvl < y.min():
            out[lvl] = np.nan
        else:
            out[lvl] = float(np.interp(lvl, y_rev, x_rev))
    return out

def plot_all_ccdf_results(
    ccdf_results,
    to_db=True,
    save_path='ccdf_comparison.png',
    column='single',              # 'single' or 'double' (only used when ieee_style=True)
    show_markers=False,
    zero_floor_for_log=1e-12,     # ONLY for plotting on log-scale (keeps CCDF behavior unchanged)
    xlim=(0, 9),
    ylim=(1e-3, 0.7),
    cmap_name="coolwarm",         # cold->hot: 'coolwarm' is intuitive; 'plasma' also good
    add_colorbar=True,            # show colorbar of rms_ds mapping
):
    """
    Plot CCDF comparison across algorithms, with curve color mapped to numeric rms_ds.

    Colors:
      - low rms_ds  -> cold colors
      - high rms_ds -> hot colors

    We extract rms_ds from curve_label text like: "... (rms_ds=3.14)".
    """


    # ---------------------------
    # Helper: extract numeric rms_ds from curve_label
    # ---------------------------
    def _extract_rms(curve_label):
        # expects something like "b12 (rms_ds=3.14)" or " ... rms_ds=3.14 ..."
        m = re.search(r"rms_ds\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", str(curve_label))
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return np.nan
        return np.nan
    def remove_outliers_percentile(data, low=5, high=95):
        lo = np.percentile(data, low)
        hi = np.percentile(data, high)
        return [x for x in data if lo <= x <= hi]
    # ---------------------------
    # Collect all rms values across ALL curves (all algs) to build one shared normalization
    # ---------------------------
    all_rms = []
    for _, result in ccdf_results.items():
        for curve_label in result["ccdf_curves"].keys():
            v = _extract_rms(curve_label)
            if np.isfinite(v):
                all_rms.append(v)
    all_rms = remove_outliers_percentile(all_rms, low=5, high=90)
    # If we can't extract any rms values, we still plot, just without rms-based coloring
    have_rms = (len(all_rms) > 0)
    if have_rms:
        vmin = float(np.min(all_rms))
        vmax = float(np.max(all_rms))
        if vmax <= vmin:
            # avoid division by zero: treat as constant
            have_rms = False
    cmap = plt.get_cmap(cmap_name)

    def _color_for_label(curve_label):
        if not have_rms:
            return None  # matplotlib default cycle
        v = _extract_rms(curve_label)
        if not np.isfinite(v):
            return None
        t = (v - vmin) / (vmax - vmin)  # 0..1
        t = float(np.clip(t, 0.0, 1.0))
        return cmap(t)

    plt.figure(figsize=(10, 8))
    for alg_name, result in ccdf_results.items():
        x_plot = result["x_plot"]
        for curve_label, ccdf in result["ccdf_curves"].items():
            y = np.asarray(ccdf)
            y_plot = np.where(y <= 0, zero_floor_for_log, y)

            marker = CCDF_marker_dict[alg_name]
            label_name = CCDF_label_name_dict[alg_name]
            color = _color_for_label(curve_label)
            if alg_name == 'OFDM Model':
                plt.step(
                    x_plot,
                    y_plot,
                    where='post',
                    # label=f"{label_name} | {curve_label}",
                    label=f"{label_name}",
                    marker=marker,
                    markersize=13,
                    markevery=0.08,
                    color='g',
                    zorder=10,
                    linewidth=4
                )
            elif alg_name == 'SC/FDE Model':
                plt.step(
                    x_plot,
                    y_plot,
                    where='post',
                    # label=f"{label_name} | {curve_label}",
                    label=f"{label_name}",
                    marker=marker,
                    markersize=13,
                    markevery=0.08,
                    color='m',
                    zorder=10,
                    linewidth=4
                )
            elif alg_name == 'E2EWL MP Model':
                plt.step(
                    x_plot,
                    y_plot,
                    where='post',
                    # label=f"{label_name} | {curve_label}",
                    label=f"{label_name}",
                    marker=marker,
                    markersize=13,
                    markevery=0.08,
                    color='k',
                    zorder=10,
                    linewidth=4
                )                
            elif alg_name == 'qQ Method Model':
                plt.step(
                    x_plot,
                    y_plot,
                    where='post',
                    # label=f"{label_name} | {curve_label}",
                    # label=f"{label_name}",
                    marker=marker,
                    markersize=10,
                    markevery=0.08,
                    color=color
                )
            if show_markers and "papr_markers" in result:
                for lvl, x_m in result["papr_markers"].get(curve_label, {}).items():
                    if np.isfinite(x_m):
                        plt.plot([x_m], [lvl], 'o', markersize=4, color=color)

    plt.xlabel(r'$PAR_0$ [dB]' if to_db else r'$PAR_0$',fontsize=20)
    plt.ylabel(r'CCDF $\; P\!\left(\frac{p(t)}{E[p(t)]} > PAR_0\right)$',fontsize=20)
    # plt.title('Empreical CCDF',fontsize=20)
    plt.grid(True, linestyle=':', linewidth=0.6)
    plt.xscale('linear' if to_db else 'log')
    plt.yscale('log')
    plt.ylim(list(ylim))
    plt.xlim(list(xlim))
    plt.tick_params(axis='both', labelsize=12)    
    plt.tight_layout()

    # legend less intrusive
    plt.legend(frameon=False,fontsize=20)

    # optional colorbar
    if add_colorbar and have_rms:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        ax = plt.gca()   # get current axes
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label(r'Delay Spread [rms] (low $\rightarrow$ high)',fontsize=20)

    plt.savefig(save_path, dpi=600)
    plt.close()
    return

def plot_all_ccdf_results_plotly(
    ccdf_results,
    to_db=True,
    html_path="ccdf_comparison.html",
    title="CCDF Comparison",
    show_markers=False,
    papr_marker_symbol="circle",
    ofdm_marker_symbol="star",
    zero_floor_for_log=1e-12,   # plotting-only floor for log axis
    x_range=(0, 15),            # matches your matplotlib default
    y_range=(1e-5, 1.2),
):
    """
    Interactive Plotly CCDF plot saved as a self-contained HTML file.

    Works with:
      ccdf_results = { alg_name: result_from_emprical_ccdf_plotter }

    Notes:
      - CCDF behavior is unchanged.
      - For log-y, zeros are floored to `zero_floor_for_log` for display.
      - Legend supports click-to-hide/show; double-click isolates one trace.
    """

    fig = go.Figure()

    # Add CCDF curves
    for alg_name, result in ccdf_results.items():
        x_plot = np.asarray(result["x_plot"])

        is_ofdm = (str(alg_name).lower() == "ofdm")

        for curve_label, ccdf in result["ccdf_curves"].items():
            y = np.asarray(ccdf)
            y_plot = np.where(y <= 0, zero_floor_for_log, y)

            trace_name = f"{alg_name} | {curve_label}"

            # Show star markers for OFDM, otherwise no markers by default
            mode = "lines"
            marker = None
            if is_ofdm:
                mode = "lines+markers"
                marker = dict(symbol=ofdm_marker_symbol, size=7)

            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    mode=mode,
                    name=trace_name,
                    line=dict(shape="hv"),  # step-like, similar to plt.step(where='post')
                    marker=marker,
                    hovertemplate="x=%{x:.3f}<br>CCDF=%{y:.3e}<extra></extra>",
                )
            )

            # Optional PAPR markers (if present in result)
            if show_markers and ("papr_markers" in result):
                markers_for_curve = result["papr_markers"].get(curve_label, {})
                for lvl, x_m in markers_for_curve.items():
                    if x_m is None:
                        continue
                    try:
                        x_m = float(x_m)
                    except Exception:
                        continue
                    if np.isfinite(x_m) and np.isfinite(lvl):
                        fig.add_trace(
                            go.Scatter(
                                x=[x_m],
                                y=[max(float(lvl), zero_floor_for_log)],
                                mode="markers+text",
                                name=f"{trace_name} marker",
                                marker=dict(symbol=papr_marker_symbol, size=8),
                                text=[f"{x_m:.2f}"],
                                textposition="top right",
                                showlegend=False,
                                hovertemplate="x=%{x:.3f}<br>CCDF=%{y:.3e}<extra></extra>",
                            )
                        )

    # Axes + layout
    fig.update_layout(
        title=title,
        xaxis_title=("Power / Avg Power [dB]" if to_db else "Power / Avg Power"),
        yaxis_title="CCDF  P(Power / Avg Power > x)",
        template="plotly_white",
        legend=dict(
            title="Click to hide/show",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=70, r=30, t=60, b=60),
    )

    # scales/ranges
    fig.update_yaxes(type="log", range=[np.log10(y_range[0]), np.log10(y_range[1])])
    if to_db:
        fig.update_xaxes(type="linear", range=list(x_range))
    else:
        fig.update_xaxes(type="log")  # if you truly want log x when not dB

    # Write HTML (self-contained)
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)
    return html_path

if __name__ == "__main__":
    # Example: 1000 signals, each with 128 samples, random complex Gaussian
    batch_size = 1000
    num_samples = 128
    x_real = tf.random.normal((batch_size, num_samples))
    x_imag = tf.random.normal((batch_size, num_samples))
    x = tf.complex(x_real, x_imag)

    emprical_ccdf_plotter(x, bins=2000, title='CCDF of instantaneous power', save_fig=True)
    emprical_papr(x)