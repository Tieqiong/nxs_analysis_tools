"""Reciprocal metadata, lazy slicing, and physical metric regressions."""

from itertools import permutations

import h5py
import numpy as np
import pytest
from nexusformat.nexus import NXdata, NXentry, NXfield, NXroot, NXsample, nxload

from nxs_analysis_tools.reciprocal import (
    physical_mesh, reciprocal_axis_coordinates, reciprocal_slice,
    resolve_nxdata, resolve_reciprocal_axes, resolve_unit_cell,
)


def make_data(order='HKL', aliases=False, edges=False, masked=False):
    centers = {'H': np.array([-2., 0., 2.]),
               'K': np.array([-1.5, -.5, .5, 1.5]),
               'L': np.arange(-2., 3.)}
    canonical = np.indices((3, 4, 5))
    values = canonical[0] * 100 + canonical[1] * 10 + canonical[2]
    if masked:
        values = np.ma.array(values, mask=values == 112)
    values = values.transpose(tuple('HKL'.index(a) for a in order))
    axes = []
    for a in order:
        x = centers[a]
        if edges:
            step = x[1] - x[0]
            x = np.r_[x - step / 2, x[-1] + step / 2]
        name = 'Q' + a.lower() if aliases else a
        axes.append(NXfield(x, name=name))
    return NXdata(NXfield(values, name='intensity'), axes)


@pytest.mark.parametrize('order', [''.join(p) for p in permutations('HKL')])
@pytest.mark.parametrize('aliases', [False, True])
@pytest.mark.parametrize('plane', ['HK', 'HL', 'KL'])
def test_native_dimension_permutations(order, aliases, plane):
    data = make_data(order, aliases)
    mapping = resolve_reciprocal_axes(data)
    assert mapping == {a: order.index(a) for a in 'HKL'}
    expected = reciprocal_slice(make_data(), plane, 0)
    result = reciprocal_slice(data, plane, 0)
    np.testing.assert_array_equal(result.values, expected.values)
    np.testing.assert_array_equal(result.x, expected.x)
    np.testing.assert_array_equal(result.y, expected.y)
    assert result.fixed_axis == next(a for a in 'HKL' if a not in plane)


def test_integer_coordinate_is_physical_and_reversed_plane():
    result = reciprocal_slice(make_data(), 'HK', 0)
    assert result.actual_coord == 0
    np.testing.assert_array_equal(result.values[0], [2, 102, 202])
    np.testing.assert_array_equal(reciprocal_slice(make_data(), 'KH', 0).values, result.values.T)


def test_explicit_edges_and_descending_coordinates():
    data = make_data(edges=True)
    centers, edges = reciprocal_axis_coordinates(data, 'L')
    np.testing.assert_array_equal(centers, [-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(edges, [-2.5, -1.5, -.5, .5, 1.5, 2.5])
    data['L'] = NXfield(edges[::-1])
    result = reciprocal_slice(data, 'HK', .5)
    # Shared boundary belongs to the higher-coordinate bin despite descending order.
    assert result.actual_coord == 1
    assert result.fixed_edges == (.5, 1.5)
    np.testing.assert_array_equal(result.values[0], [1, 101, 201])
    assert reciprocal_slice(data, coord=-2.5).actual_coord == -2
    assert reciprocal_slice(data, coord=2.5).actual_coord == 2


def test_nonuniform_axis_edges_preserved():
    data = make_data()
    data['H'] = NXfield([-3., -2., 1., 5.])
    centers, edges = reciprocal_axis_coordinates(data, 'H')
    np.testing.assert_array_equal(centers, [-2.5, -.5, 3.])
    np.testing.assert_array_equal(edges, [-3., -2., 1., 5.])
    np.testing.assert_array_equal(reciprocal_slice(data).x_edges, edges)


@pytest.mark.parametrize('coord', [-2.5001, 2.5001, np.nan, np.inf])
def test_invalid_fixed_coordinate_rejected(coord):
    with pytest.raises(ValueError):
        reciprocal_slice(make_data(), coord=coord)


@pytest.mark.parametrize('axis_values', [[0, 0, 1], [0, 2, 1], [0, np.nan, 2], [0, 1]])
def test_invalid_axes_rejected(axis_values):
    data = make_data()
    data['H'] = NXfield(axis_values)
    with pytest.raises(ValueError):
        reciprocal_slice(data)


def test_unknown_duplicate_and_wrong_rank_axes_rejected():
    data = make_data()
    data['X'] = data['H']
    data.attrs['axes'] = ['X', 'K', 'L']
    with pytest.raises(ValueError, match='uniquely'):
        resolve_reciprocal_axes(data)
    data.attrs['axes'] = ['H', 'H', 'L']
    with pytest.raises(ValueError, match='uniquely'):
        resolve_reciprocal_axes(data)
    with pytest.raises(ValueError, match='three-dimensional'):
        resolve_reciprocal_axes(NXdata(NXfield(np.ones((2, 3))), [NXfield([0, 1], name='H'), NXfield([0, 1, 2], name='K')]))


def test_singleton_fixed_axis_and_explicit_singleton_plot_edges():
    data = NXdata(NXfield(np.ones((2, 2, 1)), name='v'),
                  [NXfield([0, 1], name='H'), NXfield([0, 1], name='K'), NXfield([2.], name='L')])
    assert reciprocal_slice(data, coord=2).actual_coord == 2
    with pytest.raises(ValueError, match='singleton'):
        reciprocal_slice(data, coord=0)
    with pytest.raises(ValueError, match='explicit bin edges'):
        reciprocal_slice(data, 'HL', 0)
    data['L'] = NXfield([1., 3.])
    assert reciprocal_slice(data, 'HL', 0).values.shape == (1, 2)


def test_mask_survives_slice_and_axis_transpose():
    section = reciprocal_slice(make_data('LKH', masked=True))
    assert np.ma.isMaskedArray(section.values)
    assert np.count_nonzero(section.values.mask) == 1
    assert section.values.mask[1, 1]


def test_disk_backed_signal_and_mask_remain_lazy(tmp_path):
    path = tmp_path / 'lazy.nxs'
    with h5py.File(path, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        data = entry.create_group('transform')
        data.attrs.update(NX_class='NXdata', signal='intensity', axes=['Ql', 'Qk', 'Qh'])
        signal = data.create_dataset('intensity', shape=(100, 80, 60), chunks=(1, 80, 60), fillvalue=7.)
        signal.attrs['mask'] = 'mask'
        mask = data.create_dataset('mask', shape=(100, 80, 60), dtype='bool', chunks=(1, 80, 60), fillvalue=False)
        mask[20, 4, 5] = True
        for name, n in [('Ql', 100), ('Qk', 80), ('Qh', 60)]:
            data.create_dataset(name, data=np.arange(n, dtype=float))
    root = nxload(path)
    data = resolve_nxdata(root)
    assert data.nxsignal._value is None
    assert data['mask']._value is None
    result = reciprocal_slice(data, 'HK', 20)
    assert result.values.shape == (80, 60)
    assert result.values.mask[4, 5]
    assert result.values[0, 0] == 7
    assert data.nxsignal._value is None
    assert data['mask']._value is None


@pytest.mark.parametrize('plane', ['HK', 'HL', 'KL'])
def test_anisotropic_triclinic_metric(plane):
    cell = (4., 8., 12., 75., 85., 120.)
    section = reciprocal_slice(make_data(), plane)
    x, y = physical_mesh(section, cell)
    dx = np.array([x[0, 1] - x[0, 0], y[0, 1] - y[0, 0]]) / np.diff(section.x_edges)[0]
    dy = np.array([x[1, 0] - x[0, 0], y[1, 0] - y[0, 0]]) / np.diff(section.y_edges)[0]
    a, b, c, alpha, beta, gamma = cell
    ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
    direct = np.array([[a*a, a*b*cg, a*c*cb], [a*b*cg, b*b, b*c*ca], [a*c*cb, b*c*ca, c*c]])
    expected = np.linalg.inv(direct)
    i, j = ['HKL'.index(axis) for axis in plane]
    np.testing.assert_allclose([dx @ dx, dy @ dy, dx @ dy], [expected[i,i], expected[j,j], expected[i,j]])
    assert x.shape == (len(section.y) + 1, len(section.x) + 1)


def test_unequal_oblique_basis_vectors_and_no_two_pi():
    section = reciprocal_slice(make_data())
    x, y = physical_mesh(section, (4, 8, 10, 90, 90, 120))
    dx = np.array([x[0,1] - x[0,0], y[0,1] - y[0,0]]) / 2
    dy = np.array([x[1,0] - x[0,0], y[1,0] - y[0,0]])
    assert np.linalg.norm(dx) == pytest.approx(1 / (4 * np.sin(np.pi / 3)))
    assert np.linalg.norm(dy) == pytest.approx(np.linalg.norm(dx) / 2)
    assert dx @ dy / (np.linalg.norm(dx) * np.linalg.norm(dy)) == pytest.approx(.5)
    x2, y2 = physical_mesh(reciprocal_slice(make_data(), coord=1), (4,8,10,90,90,120))
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)


@pytest.mark.parametrize('cell', [(0,1,1,90,90,90), (1,1,1,0,90,90), (1,1,1,10,10,170), (1,1,1,90,90,np.nan), [1,2,3]])
def test_invalid_cells_rejected(cell):
    with pytest.raises(ValueError):
        resolve_unit_cell(make_data(), cell)


def test_cell_override_and_native_metadata():
    cell = (4.,8.,12.,90.,90.,120.)
    data = make_data()
    assert resolve_unit_cell(data) is None
    assert resolve_unit_cell(data, '4,8;12 90,90,120') == cell
    root = NXroot(NXentry(data, NXsample(unit_cell=cell)))
    assert resolve_unit_cell(root.entry.data) == cell
    assert resolve_unit_cell(root.entry.data, (1,1,1,90,90,90))[0] == 1
    del root.entry.sample['unit_cell']
    root.entry.sample['unit_cell_abc'] = cell[:3]
    with pytest.raises(ValueError, match='Incomplete'):
        resolve_unit_cell(root.entry.data)
    root.entry.sample['unit_cell_alphabetagamma'] = cell[3:]
    assert resolve_unit_cell(root.entry.data) == cell
    del root.entry['sample']
    root.entry['sample'] = NXsample(**dict(zip(['unitcell_' + s for s in ('a','b','c','alpha','beta','gamma')], cell)))
    assert resolve_unit_cell(root.entry.data) == cell


def test_nxdata_resolution_default_and_ambiguity():
    root = NXroot(NXentry(make_data()))
    assert resolve_nxdata(root) is root.entry.data
    root.entry['transform'] = make_data('LKH', True)
    with pytest.raises(ValueError, match='Ambiguous'):
        resolve_nxdata(root)
    root.attrs['default'] = 'entry'
    root.entry.attrs['default'] = 'transform'
    assert resolve_nxdata(root) is root.entry.transform
    assert resolve_nxdata(root.entry) is root.entry.transform
    assert resolve_nxdata(root.entry.data) is root.entry.data
    root.entry.attrs['default'] = 'missing'
    with pytest.raises(ValueError, match='Ambiguous'):
        resolve_nxdata(root)


def test_axis_indices_metadata_validated():
    data = make_data('LKH', True)
    for dim, name in enumerate(('Ql', 'Qk', 'Qh')):
        data.attrs[name + '_indices'] = [dim]
    assert resolve_reciprocal_axes(data) == {'H': 2, 'K': 1, 'L': 0}
    data.attrs['Qh_indices'] = [0]
    with pytest.raises(ValueError, match='indices must agree'):
        resolve_reciprocal_axes(data)


def test_external_nxlink_proxy_keeps_metadata_and_lazy_signal(tmp_path):
    from nexusformat.nexus import NXlink
    source_path = tmp_path / 'source.nxs'
    with h5py.File(source_path, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        data = entry.create_group('data')
        data.attrs.update(NX_class='NXdata', signal='intensity', axes=['Ql', 'Qk', 'Qh'])
        data.create_dataset('intensity', shape=(100, 80, 60), chunks=(1, 80, 60), fillvalue=7.)
        for name, n in [('Ql', 100), ('Qk', 80), ('Qh', 60)]:
            data.create_dataset(name, data=np.arange(n, dtype=float))
    proxy = NXroot(NXentry(NXlink('/entry/data', file=str(source_path), name='transform')))
    linked = resolve_nxdata(proxy)
    assert linked.nxclass == 'NXdata'
    assert resolve_reciprocal_axes(linked) == {'H': 2, 'K': 1, 'L': 0}
    assert linked.nxsignal._value is None
    result = reciprocal_slice(linked)
    assert result.values.shape == (80, 60)
    assert np.all(result.values == 7.)
    assert linked.nxsignal._value is None


def test_single_row_cell_and_explicit_metadata_units():
    cell = (4., 8., 12., 90., 90., 120.)
    root = NXroot(NXentry(make_data(), NXsample(unit_cell=[cell])))
    root.entry.sample.unit_cell.attrs['units'] = 'angstrom'
    assert resolve_unit_cell(root.entry.data) == cell
    root.entry.sample.unit_cell.attrs['units'] = 'nm'
    with pytest.raises(ValueError, match='Unsupported cell units'):
        resolve_unit_cell(root.entry.data)
    assert resolve_unit_cell(root.entry.data, cell) == cell


def test_entry_default_without_root_default_is_honored():
    root = NXroot(NXentry(make_data()))
    root.entry['transform'] = make_data('LKH', True)
    root.entry.attrs['default'] = 'transform'
    assert resolve_nxdata(root) is root.entry.transform


@pytest.mark.parametrize('units', ['rlu', 'R.L.U.', 'reciprocal lattice units',
                                   'reciprocal_lattice_unit', 'dimensionless', 'unitless', '1', ''])
def test_conventional_dimensionless_axis_units_accepted(units):
    data = make_data()
    data.H.attrs['units'] = units
    assert resolve_reciprocal_axes(data)['H'] == 0


@pytest.mark.parametrize('units', ['1/angstrom', 'nm^-1', 'm-1', 'arbitrary units'])
def test_dimensional_or_unknown_axis_units_rejected(units):
    data = make_data()
    data.H.attrs['units'] = units
    with pytest.raises(ValueError, match='dimensionless Miller coefficients'):
        reciprocal_slice(data)
    with pytest.raises(ValueError, match='dimensionless Miller coefficients'):
        resolve_nxdata(NXroot(NXentry(data)))
