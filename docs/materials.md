# Materials Guide

`califorcia` models materials through a minimal interface that supplies the dielectric response on the imaginary-frequency axis. Materials may additionally supply a magnetic response.

## Built-In Materials

The package includes predefined materials under `califorcia.materials`, including:

- `vacuum`
- `pec`
- `gold`
- `gold_drude`
- `gold_plasma`
- `aluminium`
- `teflon`
- `ethanol`
- `silica`
- `fused_silica`
- `aSiO2`
- `aAl2O3`
- `Si`
- `pdoped_Si`
- `SiC`
- `polystyrene`
- `sodalime`

Import them as:

```python
from califorcia.materials import gold, vacuum, teflon
```

## Material Interface

A material must expose:

- `materialclass`
- `epsilon(xi)`

A material may optionally expose:

- `mu(xi)`: the magnetic permeability on the imaginary-frequency axis

Example:

```python
class UserMaterial:
    def __init__(self):
        self.materialclass = "dielectric"

    def epsilon(self, xi):
        wj = 1.911e15
        cj = 1.282
        return 1.0 + cj * wj**2 / (wj**2 + xi**2)
```

The argument `xi` is the imaginary angular frequency in `rad/s`.

If `materialclass = "plasma"` is used, the material must also define the plasma frequency `wp` in rad/s.

If `mu` is defined, it must be callable as `mu(xi)`. Materials that do not define it are treated as non-magnetic, i.e. `mu = 1`.

## Supported `materialclass` Values

### `"dielectric"`

Use this for insulating or dielectric materials with a finite static permittivity.

Expected behavior:

- `epsilon(0)` is finite
- the TE zero-frequency contribution vanishes unless the material is magnetic

### `"drude"`

Use this for metals modeled with dissipation.

In practice, built-in Drude-like materials also provide:

- `wp`: plasma frequency
- `gamma`: damping rate

### `"plasma"`

Use this for dissipationless plasma models.

In practice, these materials provide:

- `wp`: plasma frequency

### `"pec"`

Perfect electric conductor handling is built into the reflection-coefficient logic.

## Magnetic Materials

A material becomes magnetic by defining `mu(xi)` alongside `epsilon(xi)`:

```python
class MagneticMaterial:
    materialclass = "dielectric"

    def epsilon(self, xi):
        return 4.0

    def mu(self, xi):
        # static permeability relaxing towards 1 above the resonance frequency
        mu0 = 5.0
        wm = 1.0e11
        return 1.0 + (mu0 - 1.0)/(1.0 + xi/wm)
```

The permeability enters in three places:

- the perpendicular wave vector inside a material becomes `kappa = sqrt(eps*mu*k0**2 + k**2)`, which sets the phase accumulated across a coating layer
- the TE Fresnel coefficient is obtained from the TM one by the electric-magnetic duality `eps <-> mu`
- the medium filling the gap contributes its own permeability, both to the wave vector between the plates and, as the material the wave is incident from, to the reflection coefficients of both plates

The physically important consequence appears at zero frequency. For non-magnetic materials the TE contribution vanishes there, whereas a static permeability contrast leaves a finite

```
rTE = (mu2 - mu1)/(mu2 + mu1)
```

which survives into the high-temperature limit. This applies to the medium as well, so a magnetic liquid between the plates is accounted for.

Because the magnetic response of real materials relaxes well below the frequencies that dominate the Casimir integral at sub-micron separations, magnetic effects usually enter almost entirely through the zero-frequency term.

Because the TM and TE coefficients carry opposite signs for predominantly electric and predominantly magnetic materials, a strongly electric plate facing a strongly magnetic plate yields a repulsive interaction, approaching Boyer's ideal-mirror result of `-7/8` times the ideal Casimir energy.

### Why The Zero-Frequency Limit Is Well Defined

The magnetic permeability of a real material stays finite as `xi -> 0`. It never diverges the way the dielectric function can: `epsilon` grows as `1/xi` for a drude material and as `1/xi**2` for a plasma material, which is why the TM coefficient needs a case distinction at `k0 = 0`.

Because `mu(0)` is finite, the zero-frequency limit of `kappa` stays finite for every `materialclass`:

| `materialclass` | `epsilon` as `xi -> 0` | `kappa(0, k)` |
| --- | --- | --- |
| `dielectric` | finite | `k` |
| `drude` | `~ 1/xi` | `k` |
| `plasma` | `~ 1/xi**2` | `sqrt(mu(0)*Kp**2 + k**2)` |

The TE coefficient therefore takes a single form in all cases and needs no case distinction of its own.

One consequence is worth noting: a magnetic drude material retains a finite TE contribution at zero frequency, whereas a non-magnetic one contributes nothing there.

## Notes On Implementation

The package evaluates dielectric response on the imaginary axis, not at real frequencies.

That means user-supplied models should implement `epsilon(xi)` directly for imaginary angular frequency input, and `mu(xi)` if the material is magnetic.

## Layered Materials

To model coatings, pass a list of materials to `matL` or `matR`.

Example:

```python
from califorcia import system
from califorcia.materials import gold, teflon, vacuum

s = system(
    300.0,
    1e-6,
    [teflon, gold],
    gold,
    vacuum,
    deltaL=[50e-9],
)
```
Ordering convention for `matL` and `matR`:

* the first entry is the coating layer closest to the intervening medium
* subsequent entries move deeper into the plate
* the last entry is the substrate half-space

Ordering convention for `deltaL` and `deltaR`:

* each entry gives the thickness of the corresponding coating layer
* the first thickness refers to the first coating layer facing the medium
* the thickness list contains only coating layers, so its length must be one less than the corresponding material list

In the example above, [teflon, gold] means a Teflon coating on a gold substrate, and deltaL=[50e-9] specifies that the Teflon layer has thickness 50 nm.

The multilayer reflection construction is recursive, so layered stacks are not limited to a fixed small number of coatings as long as the corresponding thickness list is provided.
