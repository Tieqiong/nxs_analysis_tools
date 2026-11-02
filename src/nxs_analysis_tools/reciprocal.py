"""Metadata-aware slices and physical geometry for reciprocal-space NXdata.

Reciprocal coordinates are dimensionless Miller coefficients (rlu), using the
crystallographic convention without a factor of 2*pi. Direct unit-cell lengths
are in angstroms and angles in degrees.
"""

from dataclasses import dataclass
import re
import numpy as np
from nexusformat.nexus import NeXusError

__all__ = [
    'ReciprocalSlice', 'resolve_reciprocal_axes',
    'reciprocal_axis_coordinates', 'reciprocal_slice', 'physical_mesh',
    'resolve_unit_cell', 'resolve_nxdata',
]

_ALIASES = {'h': 'H', 'qh': 'H', 'k': 'K', 'qk': 'K', 'l': 'L', 'ql': 'L'}
_RLU_UNITS = {'', '1', 'rlu', 'dimensionless', 'unitless', 'none',
              'reciprocallatticeunit', 'reciprocallatticeunits'}


@dataclass(frozen=True)
class ReciprocalSlice:
    """One reciprocal plane, oriented as rows (y), columns (x).

    ``x`` and ``y`` contain bin centers; ``x_edges`` and ``y_edges`` contain
    bin boundaries, in native ascending or descending order. ``fixed_edges``
    gives the sorted bounds of the selected bin. Coordinates are in reciprocal
    lattice units. ``values`` preserves an available signal mask.
    """

    values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    plane: str
    fixed_axis: str
    requested_coord: float
    actual_coord: float
    fixed_edges: tuple


def resolve_reciprocal_axes(data):
    """Resolve H, K, and L to native signal dimensions using NXdata axes.

    Parameters
    ----------
    data : NXdata
        Three-dimensional reciprocal-space data. Axis names must be H/K/L or
        Qh/Qk/Ql (case insensitive). Coordinates must be dimensionless Miller
        coefficients: absent units, rlu, and dimensionless labels are accepted.
        Explicit dimensional or unknown units raise an error. Array order is
        never inferred.

    Returns
    -------
    dict
        Mapping from canonical H, K, L labels to integer dimensions.

    Raises
    ------
    ValueError
        If signal rank, axis names, uniqueness, or axis lengths are invalid.
    """
    if getattr(data, 'nxclass', None) != 'NXdata':
        raise ValueError('Expected an NXdata group')
    signal = data.nxsignal
    if signal is None or len(signal.shape) != 3:
        raise ValueError('Reciprocal slicing requires a three-dimensional signal')
    axes = data.nxaxes
    if axes is None or len(axes) != 3:
        raise ValueError('NXdata must declare three reciprocal axes')
    result = {}
    for dim, axis in enumerate(axes):
        label = _ALIASES.get(str(axis.nxname).strip().lower())
        if label is None or label in result:
            raise ValueError('NXdata axes must uniquely identify H, K, L (or Qh, Qk, Ql)')
        units = axis.attrs.get('units')
        if units is not None:
            normalized_units = re.sub(r'[\s._-]+', '', str(units).strip().lower())
            if normalized_units not in _RLU_UNITS:
                raise ValueError(f'{label} axis units {units!s} are unsupported; expected dimensionless Miller coefficients (rlu)')
        indices = data.attrs.get(str(axis.nxname) + '_indices')
        if indices is not None:
            indices = np.atleast_1d(getattr(indices, 'nxvalue', indices))
            if indices.shape != (1,) or indices[0] != dim:
                raise ValueError(f'{label} axis indices must agree with its one-dimensional NXdata axes position')
        if len(axis.shape) != 1 or axis.shape[0] not in (signal.shape[dim], signal.shape[dim] + 1):
            raise ValueError(f'{label} axis length must match its signal dimension or its bin edges')
        if signal.shape[dim] == 0:
            raise ValueError(f'{label} signal dimension is empty')
        result[label] = dim
    return {label: result[label] for label in 'HKL'}


def reciprocal_axis_coordinates(data, axis='H'):
    """Return validated centers and edges for one named reciprocal axis.

    Parameters
    ----------
    data : NXdata
        Three-dimensional data with named reciprocal axes.
    axis : {'H', 'K', 'L'}, optional
        Canonical axis name, case insensitive.

    Returns
    -------
    centers : ndarray
        Native-order bin centers.
    edges : ndarray or None
        Explicit native boundaries, or midpoint-inferred boundaries for center
        axes. A singleton center has no inferable width and returns None.

    Notes
    -----
    Nonuniform and descending axes are supported. Center-axis endpoints are
    extrapolated by half the nearest center spacing. Explicit edge axes are
    authoritative. Axes must be finite and strictly monotonic.
    """
    label = str(axis).upper()
    mapping = resolve_reciprocal_axes(data)
    if label not in mapping:
        raise ValueError('axis must be H, K, or L')
    dim = mapping[label]
    values = np.asarray(data.nxaxes[dim].nxdata, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{label} coordinates must be finite')
    delta = np.diff(values)
    if len(delta) and not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError(f'{label} coordinates must be strictly monotonic')
    n = data.nxsignal.shape[dim]
    if len(values) == n + 1:
        return values[:-1] / 2 + values[1:] / 2, values.copy()
    if n == 1:
        return values.copy(), None
    edges = np.empty(n + 1, dtype=float)
    edges[1:-1] = values[:-1] / 2 + values[1:] / 2
    edges[0] = values[0] - delta[0] / 2
    edges[-1] = values[-1] + delta[-1] / 2
    if not np.all(np.isfinite(edges)):
        raise ValueError(f'{label} inferred bin edges must be finite')
    return values.copy(), edges


def _coordinate_index(centers, edges, coord, label):
    if edges is None:
        if coord != centers[0]:
            raise ValueError(f'{label} singleton axis only supports coordinate {centers[0]}')
        return 0, (float(centers[0]), float(centers[0]))
    ascending = edges[-1] > edges[0]
    ordered = edges if ascending else edges[::-1]
    if coord < ordered[0] or coord > ordered[-1]:
        raise ValueError(f'{label} coordinate {coord} is outside [{ordered[0]}, {ordered[-1]}]')
    index = min(int(np.searchsorted(ordered, coord, side='right')) - 1, len(centers) - 1)
    if not ascending:
        index = len(centers) - 1 - index
    return index, tuple(sorted((float(edges[index]), float(edges[index + 1]))))


def reciprocal_slice(data, plane='HK', coord=0.0):
    """Read one physical-coordinate slice without materializing the 3D signal.

    Parameters
    ----------
    data : NXdata
        Three-dimensional reciprocal-space data.
    plane : {'HK', 'HL', 'KL'}, optional
        Horizontal then vertical axes. Reversed pairs are also accepted.
    coord : float, optional
        Physical coordinate on the remaining axis, even when passed as an int.

    Returns
    -------
    ReciprocalSlice
        The selected bin, oriented with vertical-axis rows and horizontal-axis
        columns, together with the requested and actual fixed coordinates.

    Notes
    -----
    Selection uses containing bin edges. Internal boundaries belong to the bin
    on the higher-coordinate side, regardless of axis direction. Both outer
    endpoints are accepted; out-of-range coordinates raise ValueError. A
    singleton plotting axis requires explicit bin edges rather than an assumed
    width. Only the selected signal and mask slabs are read from disk.
    """
    plane = str(plane).upper()
    if len(plane) != 2 or len(set(plane)) != 2 or any(a not in 'HKL' for a in plane):
        raise ValueError('plane must contain two distinct H, K, L axes')
    coord = float(coord)
    if not np.isfinite(coord):
        raise ValueError('Slice coordinate must be finite')
    mapping = resolve_reciprocal_axes(data)
    fixed_axis = next(a for a in 'HKL' if a not in plane)
    x, x_edges = reciprocal_axis_coordinates(data, plane[0])
    y, y_edges = reciprocal_axis_coordinates(data, plane[1])
    fixed, fixed_edges = reciprocal_axis_coordinates(data, fixed_axis)
    if x_edges is None or y_edges is None:
        raise ValueError('A singleton plotting axis requires explicit bin edges')
    index, bounds = _coordinate_index(fixed, fixed_edges, coord, fixed_axis)
    selection = [slice(None)] * 3
    selection[mapping[fixed_axis]] = index
    selection = tuple(selection)
    signal = data.nxsignal
    values = np.asanyarray(signal[selection].nxdata)
    # NXfield slices can lose a sibling-mask reference; slice that mask too.
    mask_name = signal.attrs.get('mask')
    if mask_name is not None:
        mask = np.asarray(signal.nxgroup[str(mask_name)][selection].nxdata, dtype=bool)
        values = np.ma.array(values, mask=np.ma.getmaskarray(values) | mask, copy=False)
    remaining = [dim for dim in range(3) if dim != mapping[fixed_axis]]
    order = (remaining.index(mapping[plane[1]]), remaining.index(mapping[plane[0]]))
    values = values.transpose(order)
    return ReciprocalSlice(values, x, y, x_edges, y_edges, plane, fixed_axis,
                           coord, float(fixed[index]), bounds)


def _validated_cell(unit_cell):
    if isinstance(unit_cell, str):
        unit_cell = re.split(r'[,;\s]+', unit_cell.strip())
    try:
        cell = np.asarray(unit_cell, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Unit cell must contain a, b, c, alpha, beta, gamma') from exc
    if cell.shape != (6,) or not np.all(np.isfinite(cell)):
        raise ValueError('Unit cell must contain six finite values')
    lengths, angles = cell[:3], cell[3:]
    if np.any(lengths <= 0) or np.any(angles <= 0) or np.any(angles >= 180):
        raise ValueError('Unit-cell lengths must be positive and angles between 0 and 180 degrees')
    ca, cb, cg = np.cos(np.deg2rad(angles))
    normalized = np.array([[1, cg, cb], [cg, 1, ca], [cb, ca, 1]])
    try:
        np.linalg.cholesky(normalized)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Unit cell must have a positive-definite metric') from exc
    return tuple(float(v) for v in cell), normalized * np.outer(lengths, lengths)


def _cell_metadata_values(field, angle=False):
    units = field.attrs.get('units')
    allowed = ('degree', 'degrees', 'deg') if angle else ('angstrom', 'angstroms', 'a', '\u00e5')
    if units is not None and str(units).strip().lower() not in allowed:
        raise ValueError(f'Unsupported cell units {units!s}; supply a cell in angstroms and degrees')
    values = np.asarray(field.nxdata)
    # NXsample/unit_cell may store one row per cell; only one cell is unambiguous.
    if values.shape == (1, 6):
        values = values[0]
    return np.atleast_1d(values)


def resolve_unit_cell(data, unit_cell=None):
    """Resolve an explicit cell or known NXsample cell metadata.

    Parameters
    ----------
    data : NXdata or NXentry
        Data whose ancestor entries may contain a ``sample`` group.
    unit_cell : sequence of float or str, optional
        Override as a, b, c, alpha, beta, gamma. Strings may use commas,
        semicolons, or whitespace. Lengths are angstroms, angles degrees.

    Returns
    -------
    tuple or None
        Validated six-parameter cell, or None when no cell metadata is present.

    Notes
    -----
    Recognized NXsample fields are ``unit_cell``, the pair ``unit_cell_abc``
    and ``unit_cell_alphabetagamma``, or the six NXRefine ``unitcell_a`` through
    ``unitcell_gamma`` fields. Their conventional units are angstroms/degrees.
    Missing or malformed members, or explicit units other than angstroms and
    degrees, raise an error. A single-row ``unit_cell`` array is supported;
    multiple cells require an explicit choice. An explicit cell takes
    precedence. No cubic cell is assumed.
    """
    if unit_cell is not None:
        return _validated_cell(unit_cell)[0]
    current = data
    while current is not None:
        if getattr(current, 'nxclass', None) in ('NXentry', 'NXsubentry') and 'sample' in current:
            sample = current['sample']
            if 'unit_cell' in sample:
                return _validated_cell(_cell_metadata_values(sample['unit_cell']))[0]
            paired = ('unit_cell_abc', 'unit_cell_alphabetagamma')
            scalar = tuple('unitcell_' + p for p in ('a', 'b', 'c', 'alpha', 'beta', 'gamma'))
            for names in (paired, scalar):
                present = [name in sample for name in names]
                if any(present):
                    if not all(present):
                        raise ValueError('Incomplete NXsample unit-cell metadata')
                    values = np.concatenate([
                        _cell_metadata_values(sample[name], angle=(i >= (1 if names == paired else 3)))
                        for i, name in enumerate(names)
                    ])
                    return _validated_cell(values)[0]
        current = getattr(current, 'nxgroup', None)
    return None


def physical_mesh(section, unit_cell):
    """Map slice bin corners into a Cartesian plane using the reciprocal metric.

    Parameters
    ----------
    section : ReciprocalSlice
        Slice returned by :func:`reciprocal_slice`.
    unit_cell : sequence of float or str
        a, b, c in angstroms and alpha, beta, gamma in degrees.

    Returns
    -------
    X, Y : ndarray
        Corner grids with shape (ny + 1, nx + 1), in inverse angstroms, suitable
        for ``pcolormesh(X, Y, section.values)`` with equal Cartesian aspect.

    Notes
    -----
    The first plane basis vector lies along +X; the second lies in the upper
    half-plane. The origin is the point with both free reciprocal coordinates
    zero at the selected fixed coordinate. Thus these are plane-local
    Cartesian distances, without a fixed-axis translation. The full inverse
    direct-cell metric sets lengths and skew; no 2*pi factor is applied.
    """
    _, metric = _validated_cell(unit_cell)
    reciprocal = np.linalg.inv(metric)
    i, j = ('HKL'.index(axis) for axis in section.plane)
    scale_x = np.sqrt(reciprocal[i, i])
    projection = reciprocal[i, j] / scale_x
    scale_y = np.sqrt(reciprocal[j, j] - projection**2)
    u, v = np.meshgrid(section.x_edges, section.y_edges)
    return scale_x * u + projection * v, scale_y * v


def resolve_nxdata(root):
    """Find a reciprocal 3D NXdata using defaults or conventional entry paths.

    Parameters
    ----------
    root : NXroot, NXentry, or NXdata
        Open NeXus object. Signals remain lazy.

    Returns
    -------
    NXdata
        Suitable data selected by a valid NeXus ``default`` chain, or the sole
        suitable ``data``/``transform`` child of a top-level NXentry.

    Raises
    ------
    ValueError
        If no suitable data exists, or conventional paths are ambiguous.
    """
    failures = {}

    def suitable(candidate):
        try:
            resolve_reciprocal_axes(candidate)
            return True
        except (ValueError, KeyError, AttributeError, TypeError, NeXusError) as exc:
            if getattr(candidate, 'nxclass', None) == 'NXdata':
                failures[candidate.nxpath] = str(exc)
            return False

    if getattr(root, 'nxclass', None) == 'NXdata':
        resolve_reciprocal_axes(root)
        return root
    def via_defaults(current):
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if suitable(current):
                return current
            default = getattr(current, 'attrs', {}).get('default')
            if default is None:
                break
            try:
                current = current[str(default)]
            except (KeyError, ValueError, NeXusError):
                break
        return None

    preferred = via_defaults(root)
    if preferred is not None:
        return preferred
    if getattr(root, 'nxclass', None) in ('NXentry', 'NXsubentry'):
        entries = [root]
    else:
        entries = [value for value in root.entries.values()
                   if getattr(value, 'nxclass', None) in ('NXentry', 'NXsubentry')]
    candidates = {}
    for entry in entries:
        preferred = via_defaults(entry)
        if preferred is not None:
            candidates[preferred.nxpath] = preferred
            continue
        for name in ('data', 'transform'):
            if name in entry and suitable(entry[name]):
                candidate = entry[name]
                candidates[candidate.nxpath] = candidate
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    if candidates:
        raise ValueError('Ambiguous reciprocal NXdata; set NeXus default metadata or select an NXdata explicitly: '
                         + ', '.join(candidates))
    details = '; '.join(f'{path}: {reason}' for path, reason in failures.items())
    raise ValueError('No suitable three-dimensional reciprocal NXdata found in defaults or entry/data, entry/transform'
                     + (f' ({details})' if details else ''))
