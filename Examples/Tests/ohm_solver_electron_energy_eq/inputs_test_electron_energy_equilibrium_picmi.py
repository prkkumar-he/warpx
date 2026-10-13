#!/usr/bin/env python3
"""Check stationary QDSMC remapping and the Biermann source in 2D and RZ.

Adapted from Prabhat Kumar's reproducing cases attached to WarpX PR #7344.
The stationary case has parallel density and temperature gradients, so neither
field should evolve. The crossed-gradient case has an independently prescribed
temperature and must generate the magnetic field predicted by Faraday's law.
Ions are held fixed to isolate the electron-energy and magnetic-field updates.
"""

import argparse

import numpy as np

from pywarpx import picmi

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", choices=["2d", "rz"], required=True)
parser.add_argument("--case", choices=["stationary", "biermann"], required=True)
args = parser.parse_args()

constants = picmi.constants
cylindrical = args.geometry == "rz"
n0, temperature_eV = 1.0e19, 10.0
amplitude = 0.2
length, cells = 1.0, 32
dt, steps = 1.0e-10, 20
kr = np.pi / length if cylindrical else 2.0 * np.pi / length
kz = 2.0 * np.pi / length
radial_coordinate = "r" if cylindrical else "x"
grid_class = picmi.CylindricalGrid if cylindrical else picmi.Cartesian2DGrid
grid = grid_class(
    number_of_cells=[cells, cells],
    lower_bound=[0.0, 0.0],
    upper_bound=[length, length],
    lower_boundary_conditions=["none" if cylindrical else "periodic", "periodic"],
    upper_boundary_conditions=["dirichlet" if cylindrical else "periodic", "periodic"],
    lower_boundary_conditions_particles=[
        "none" if cylindrical else "periodic",
        "periodic",
    ],
    upper_boundary_conditions_particles=[
        "reflecting" if cylindrical else "periodic",
        "periodic",
    ],
    warpx_max_grid_size=16,
    warpx_blocking_factor=8,
)
solver = picmi.HybridPICSolver(
    grid=grid,
    Te=temperature_eV,
    n0=n0,
    gamma=5.0 / 3.0,
    n_floor=1.0e-3 * n0,
    plasma_resistivity=0.0,
    substeps=2,
    solve_electron_energy_equation=True,
)
# In RZ the analytic particle distribution is evaluated in Cartesian coordinates.
radius = "sqrt(x*x+y*y)" if cylindrical else "x"
ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass="m_p",
    warpx_do_not_push=True,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=f"{n0}*(1 + {amplitude}*cos({kr}*{radius}))"
    ),
)
sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=steps + 1,
    particle_shape="linear",
    verbose=0,
    warpx_current_deposition_algo="direct",
    warpx_use_filter=False,
    warpx_serialize_initial_conditions=True,
)
sim.add_species(
    ions,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=[4, 8, 4] if cylindrical else [4, 4], grid=grid
    ),
)

# The first step initializes Te from the polytropic closure. Replace it with an
# independent profile using the public field interface, including periodic ghosts.
sim.step(1)
temperature = sim.fields.get("hybrid_electron_temperature_fp", level=0)
magnetic = sim.fields.get("Bfield_fp", dir="theta" if cylindrical else "y", level=0)
r, z = np.meshgrid(
    temperature.mesh(radial_coordinate), temperature.mesh("z"), indexing="ij"
)
profile = np.cos(2.0 * kr * r) if args.case == "stationary" else np.cos(kz * z)
initial_temperature = temperature_eV * (1.0 + amplitude * profile)
temperature[:, :] = initial_temperature * constants.q_e / constants.kb
temperature.fill_boundary(sim.extension.warpx.Geom(lev=0).periodicity())
initial_magnetic = magnetic[:, :].copy()

sim.step(steps)
final_temperature = temperature[:, :] * constants.kb / constants.q_e
delta_magnetic = magnetic[:, :] - initial_magnetic
temperature_error = np.max(np.abs(final_temperature - initial_temperature))
print(f"Maximum temperature change: {temperature_error:.6e} eV")
print(f"Maximum generated magnetic field: {np.max(np.abs(delta_magnetic)):.6e} T")

if args.case == "stationary":
    # A marker at rest should return its original nodal entropy. This checks
    # the entire nodal field, including the axis and non-periodic radial face.
    assert temperature_error < 1.0e-4 * temperature_eV, "Stationary Te changed"
    assert np.max(np.abs(delta_magnetic)) < 1.0e-12, "Parallel gradients generated B"
else:
    r, z = np.meshgrid(
        magnetic.mesh(radial_coordinate), magnetic.mesh("z"), indexing="ij"
    )
    # dB_y/dt = (1/ne) dTe[eV]/dz dne/dx; B_theta has the same sign in RZ.
    expected = (
        dt
        * steps
        * temperature_eV
        * amplitude**2
        * kr
        * kz
        * np.sin(kr * r)
        * np.sin(kz * z)
        / (1.0 + amplitude * np.cos(kr * r))
    )
    # Exclude the RZ boundary stencil: the analytic profile describes the bulk,
    # whereas the outer conducting wall sets its own field boundary condition.
    interior = (slice(2, -2), slice(None)) if cylindrical else (...,)
    relative_error = np.linalg.norm(
        (delta_magnetic - expected)[interior]
    ) / np.linalg.norm(expected[interior])
    print(
        f"Maximum B in the comparison region: {np.max(np.abs(delta_magnetic[interior])):.6e} T"
    )
    print(f"Biermann magnetic-field relative L2 error: {relative_error:.6e}")
    assert relative_error < 0.05, "Magnetic field does not match the Biermann source"

sim.finalize()
