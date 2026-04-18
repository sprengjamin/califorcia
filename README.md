# califorcia

`califorcia` is a Python package for computing Casimir-Lifshitz interactions between laterally infinite planar bodies using Lifshitz theory.

It supports:

- Free energy per unit area
- Pressure
- Pressure gradient
- Zero-temperature and finite-temperature calculations
- Homogeneous half-spaces and coated planes
- Built-in and user-defined material response models
- Recursive multilayer reflection construction for layered coatings
- **Specialized support for electrolyte media and the longitudinal scattering channel**

<p align="center">
  <img src="images/casimir_effect.jpg" height="60%" width="60%">
</p>

## What The Package Models

The main object in the package is `califorcia.system`, which represents two parallel planes separated by a medium:

- left plane
- right plane
- intervening medium
- separation distance `d`
- temperature `T`
- optional coating layers on either plane

All returned quantities are in SI units and are given per unit area.

## Installation

From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

- Windows:

```bash
.venv\Scripts\activate
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

Install the package:

```bash
pip install .
```

For development:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Compute the free energy and pressure between two gold half-spaces in vacuum at `T = 300 K` and `d = 1 um`:

```python
from califorcia import system
from califorcia.materials import gold, vacuum

T = 300.0
d = 1.0e-6

s = system(T, d, gold, gold, vacuum)

print("energy:", s.energy())
print("pressure:", s.pressure())
```

## Package Overview

### Public API

- `from califorcia import system`
- `system.energy(...)`
- `system.pressure(...)`
- `system.pressuregradient(...)`
- `system.calculate(observable, ...)`

The main solver methods expose `epsrel` and `epsabs` to control numerical tolerances.

### Built-in Materials

The repository ships predefined materials in [`califorcia/materials`](califorcia/materials), including:

- `vacuum`
- `gold`, `gold_drude`, `gold_plasma`
- `pec`
- `ethanol`
- `teflon`
- `silica`, `fused_silica`, `aSiO2`
- `aluminium`, `aAl2O3`
- `Si`, `pdoped_Si`, `SiC`
- `polystyrene`
- `sodalime`
- `salt_water`

## Documentation

Additional project documentation is available in [`docs/`](docs):

- [Getting started](docs/getting-started.md)
- [API reference](docs/api.md)
- [Materials guide](docs/materials.md)
- [Development notes](docs/development.md)

## Examples

Runnable examples are provided in [`examples/`](examples):

- [`example1.py`](examples/example1.py): two gold half-spaces in vacuum
- [`example2.py`](examples/example2.py): custom user-defined material
- [`example3.py`](examples/example3.py): coated surface
- [`example4.py`](examples/example4.py): varying plasma frequency

## Testing

Run the test suite from the repository root:

```bash
pytest
```

Current regression tests are in [`tests/test_results.py`](tests/test_results.py).

GitHub Actions continuous integration is configured in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and runs the test suite automatically on pushes to `master` and on pull requests.

## Citation And Scientific Use

If you use this package in research, it is a good idea to cite:

- the repository itself
- the Lifshitz-theory references that define your physical model
- the specific material models you rely on, where relevant

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE).
