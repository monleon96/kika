from typing import Any, Mapping, Optional, Sequence, Union, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from mcnpy.serpent.sens import SensitivityFile
from mcnpy._plot_settings import setup_plot_style, format_axes, finalize_plot


IndexLike = Union[int, str]
IndexSeq = Sequence[IndexLike]


def _as_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]


def _material_indices(sf: SensitivityFile, materials: Optional[Union[IndexLike, IndexSeq]]) -> List[int]:
    if materials is None:
        return list(range(sf.n_materials))
    out = []
    for m in _as_list(materials):
        out.append(sf._material_index(m))
    return out


def _nuclide_indices(sf: SensitivityFile, nuclides: Optional[Union[IndexLike, IndexSeq]]) -> List[int]:
    if nuclides is None:
        return list(range(sf.n_nuclides))
    out = []
    for n in _as_list(nuclides):
        out.append(sf._nuclide_index(n))
    return out


def _collect_perturbations(
    sf: SensitivityFile,
    mt: Optional[Union[int, Sequence[int]]] = None,
    leg: Optional[Union[int, Sequence[int]]] = None,
    leg_channel: Optional[str] = None,
) -> List[int]:
    """
    Build perturbation index list from MT numbers and/or Legendre orders.
    - mt: int or list[int] of MT values (e.g., 1, 2, 102, ...). 
      Note: MT 400X are automatically recognized as Legendre moments.
    - leg: int or list[int] of Legendre orders (L), e.g., 1, 2, 3
      These are converted to MT 400X format internally.
    - leg_channel: deprecated (Legendre moments are elastic only), kept for backward compatibility
    If both mt and leg are None, returns ALL perturbations.
    """
    def _to_set(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple, np.ndarray)):
            return set(int(v) for v in x)
        return {int(x)}

    mt_set = _to_set(mt)
    leg_set = _to_set(leg)

    # Convert Legendre orders to MT 400X format and merge with mt_set
    if leg_set is not None:
        leg_as_mt = {4000 + L for L in leg_set}  # L=1 -> MT=4001, L=2 -> MT=4002, etc.
        if mt_set is not None:
            mt_set = mt_set | leg_as_mt
        else:
            mt_set = leg_as_mt

    if mt_set is None:
        return list(range(sf.n_perturbations))

    # Find perturbations matching the MT numbers (including 400X for Legendre)
    idxs: List[int] = []
    for p in sf.perturbations:
        if p.mt is not None and p.mt in mt_set:
            idxs.append(p.index)

    # Remove duplicates while preserving order
    seen = set()
    uniq = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def _default_label(sf: SensitivityFile, response: str, mi: int, zi: int, pi: int) -> str:
    mname = sf.materials[mi].name
    znum = sf.nuclides[zi].zai              # ZAI number only (cleaned format, no 'ZAI' prefix)
    plab = sf.perturbations[pi].short_label or sf.perturbations[pi].raw_label
    return f"{response} | {mname} | {znum} | {plab}"


def _clip_range(
    edges: np.ndarray,
    vals: np.ndarray,
    rel: np.ndarray,
    widths: np.ndarray,
    energy_range: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clip arrays to an energy range. Returns (edges_c, vals_c, rel_c, widths_c)."""
    if energy_range is None:
        return edges, vals, rel, widths

    emin, emax = energy_range
    if emin >= emax:
        raise ValueError("energy_range must satisfy emin < emax")

    low = edges[:-1]
    high = edges[1:]
    mask = (high > emin) & (low < emax)
    if not np.any(mask):
        return np.array([emin, emax], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    i0 = int(np.argmax(mask))
    i1 = int(len(mask) - 1 - np.argmax(mask[::-1]))

    edges_c = edges[i0:i1 + 2].astype(float).copy()
    edges_c[0] = max(edges_c[0], float(emin))
    edges_c[-1] = min(edges_c[-1], float(emax))

    vals_c = vals[i0:i1 + 1]
    rel_c = rel[i0:i1 + 1]
    widths_c = widths[i0:i1 + 1]
    return edges_c, vals_c, rel_c, widths_c


def plot_energy_sensitivity(
    sf: SensitivityFile,
    responses: Union[str, Sequence[str]],
    *,
    materials: Optional[Union[IndexLike, IndexSeq]] = None,
    nuclides: Optional[Union[IndexLike, IndexSeq]] = None,
    mt: Optional[Union[int, Sequence[int]]] = None,
    leg: Optional[Union[int, Sequence[int]]] = None,
    leg_channel: Optional[str] = None,
    per_lethargy: bool = True,
    errorbars: bool = True,
    ax: Optional[plt.Axes] = None,
    logx: bool = False,
    energy_range: Optional[Tuple[float, float]] = None,
    legend: bool = True,
    label_fmt: Optional[str] = None,
    style: str = 'default',
    figsize: Tuple[float, float] = (8, 6),
):
    """
    Step-plot energy-dependent sensitivities for one or multiple responses and selections.

    Parameters
    ----------
    sf : SensitivityFile
        Parsed sensitivity file object.
    responses : str | list[str]
        One response full-name (e.g., 'sens_ratio_BIN_0') or a list of them.
    materials, nuclides :
        Single index/name/ZAI or a list. If None -> plot all.
        - materials: int index or material name (str)
        - nuclides: int index, ZAI (int), or 'ZAI<num>' string
    mt : int | list[int] | None
        MT numbers to plot (e.g., 1 for total, 2, 102, ...). If None, not used.
    leg : int | list[int] | None
        Legendre orders to plot (e.g., 1, 2, 3). If None, not used.
    leg_channel : str | None
        Optional channel filter for legendre (e.g., 'ela'). If None, include all channels.
    per_lethargy : bool
        If True (default), values and errors are divided by lethargy widths.
    errorbars : bool
        Draw vertical error bars at bin midpoints (default True).
    ax : matplotlib.axes.Axes
        Optional axes to draw on; if None, a new figure/axes is created using setup_plot_style.
    logx : bool
        Use log scale on x-axis (default False).
    energy_range : (emin, emax) or None
        If provided, clip the plot range to [emin, emax].
    legend : bool
        Show legend (default True).
    label_fmt : str | None
        Optional legend label template. Fields: {response}, {material}, {nuclide}, {pert}.
    style : str
        Plot style key (integrates with your _plot_settings).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if isinstance(responses, str):
        resp_list = [responses]
    else:
        resp_list = list(responses)

    m_inds = _material_indices(sf, materials)
    z_inds = _nuclide_indices(sf, nuclides)
    p_inds = _collect_perturbations(sf, mt=mt, leg=leg, leg_channel=leg_channel)

    # default y label up-front (so it exists even if no lines drawn)
    y_label = "Sensitivity per lethargy" if per_lethargy else "Sensitivity"

    # Setup plot style and get axes if not provided
    plot_settings = None
    if ax is None:
        plot_settings = setup_plot_style(style=style, figsize=figsize, ax=ax)
        ax = plot_settings['ax']
        fig = plot_settings['_fig']
        colors = plot_settings['_colors']
    else:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key().get('color', ['C0'])
        fig = ax.get_figure()

    full_edges = sf.energy_grid
    full_widths = sf.lethargy_widths

    # Define line styles that will cycle for different responses
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
    
    # Counter for perturbations (to assign consistent colors)
    perturbation_counter = 0

    for resp_idx, resp in enumerate(resp_list):
        # Get line style for this response
        linestyle = linestyles[resp_idx % len(linestyles)]
        
        # Reset perturbation counter for consistent colors across responses
        temp_counter = 0
        
        for mi in m_inds:
            for zi in z_inds:
                for pi in p_inds:
                    # Access the raw sensitivity data directly using the specific perturbation index
                    sset, b = sf._locate_set_and_bin(resp)
                    vals, rel = sset.get_ed_by_index(b, mi, zi, pi, None)
                    vals = np.asarray(vals, dtype=float).reshape(-1)
                    rel = np.asarray(rel, dtype=float).reshape(-1)
                    if vals.size != full_edges.size - 1:
                        raise RuntimeError("Unexpected shape from get_energy_dependent; expected length equals number of energy bins.")

                    # Clip to energy_range (if any)
                    edges, v, r, w = _clip_range(full_edges, vals, rel, full_widths, energy_range)
                    if v.size == 0:
                        continue

                    if per_lethargy:
                        w_safe = np.where(w > 0.0, w, np.nan)
                        y = v / w_safe
                        yerr = np.abs(y) * r
                    else:
                        y = v
                        yerr = np.abs(v) * r

                    y_step = np.r_[y, y[-1]]

                    if label_fmt:
                        lbl = label_fmt.format(
                            response=resp,
                            material=sf.materials[mi].name,
                            nuclide=sf.nuclides[zi].zai,
                            pert=sf.perturbations[pi].short_label or sf.perturbations[pi].raw_label,
                        )
                    else:
                        lbl = _default_label(sf, resp, mi, zi, pi)

                    # Use consistent color based on perturbation, but different line style per response
                    color = colors[temp_counter % len(colors)]
                    ax.step(edges, y_step, where="post", label=lbl, color=color, linestyle=linestyle)

                    if errorbars:
                        mids = np.sqrt(edges[:-1] * edges[1:]) if logx else 0.5 * (edges[:-1] + edges[1:])
                        ax.errorbar(mids, y, yerr=yerr, fmt="none", linewidth=1, capsize=2, zorder=3, color=color)

                    temp_counter += 1

    ax = format_axes(
        ax=ax,
        style=style,
        use_log_scale=logx,
        is_energy_axis=True,
        x_label="Energy (MeV)",
        y_label=y_label,
        legend_loc='best' if legend else None
    )

    if energy_range is not None:
        ax.set_xlim(energy_range)

    if plot_settings is not None:
        finalize_plot(fig, plot_settings['_notebook_mode'])

    return ax
