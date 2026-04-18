# API Reference

## `califorcia.system`

Imported as:

```python
from califorcia import system
```

## Constructor

```python
system(T, d, matL, matR, matm, deltaL=[], deltaR=[])
```

Parameters:

- `T` (`float`): temperature in kelvin
- `d` (`float`): plate separation in meters
- `matL` (`object | list[object]`): left-plane material or layered stack
- `matR` (`object | list[object]`): right-plane material or layered stack
- `matm` (`object`): medium material
- `deltaL` (`list[float]`): left coating thicknesses in meters
- `deltaR` (`list[float]`): right coating thicknesses in meters

If a side is layered:

- the first material faces the medium
- the last material is the substrate half-space
- each coating layer needs a corresponding thickness entry

## Main Methods

### `system.energy(ht_limit=False, fs="psd", epsrel=1e-8, epsabs=0.0, N=None)`

Returns the Casimir free energy per unit area.

### `system.pressure(ht_limit=False, fs="psd", epsrel=1e-8, epsabs=0.0, N=None)`

Returns the Casimir pressure.

### `system.pressuregradient(ht_limit=False, fs="psd", epsrel=1e-8, epsabs=0.0, N=None)`

Returns the pressure gradient.

### `system.calculate(observable, ht_limit=False, fs="psd", epsrel=1e-8, epsabs=0.0, N=None)`

General entry point used by the convenience methods above.

Supported `observable` values:

- `"energy"`
- `"pressure"`
- `"pressuregradient"`

### `system.calculate_longitudinal(observable, epsrel=1e-8, epsabs=0.0)`

Calculates the Casimir interaction due to the longitudinal scattering channel, specifically for media with ions in solution (electrolytes).

As per [Phys. Rev. A 111, 012816 (2025)](https://doi.org/10.1103/PhysRevA.111.012816), this method only considers the zero-frequency ($n=0$) contribution, which is the only relevant term for this channel in electrolyte systems.

Supported `observable` values:

- `"energy"`
- `"pressure"`

## Numerical Options

### `ht_limit`

If `True`, only the zero-frequency contribution is returned.

### `fs`

Finite-temperature summation method:

- `"psd"`: Pade spectrum decomposition
- `"msd"`: direct Matsubara summation

### `epsrel`

Relative target accuracy for the finite-frequency sum.

### `epsabs`

Absolute target accuracy for the radial quadratures and for the zero-temperature frequency integration.

### `N`

Optional summation order. If omitted, the implementation selects it automatically.

## Return Values

All main methods return a single `float` in SI units.

The quantity is always per unit area.

## Error Conditions

The constructor raises `ValueError` for invalid physical or material inputs, including:

- coating thickness lists that do not match the number of layers
- non-positive plate separation
- negative coating thickness
- unsupported `materialclass`
- plasma materials that do not define `wp`

`calculate(...)` raises `ValueError` for unsupported observables or unsupported frequency-summation methods.

## Minimal Example

```python
from califorcia import system
from califorcia.materials import gold, vacuum

s = system(300.0, 1e-6, gold, gold, vacuum)

print(s.energy())
print(s.pressure())
print(s.pressuregradient())
```
