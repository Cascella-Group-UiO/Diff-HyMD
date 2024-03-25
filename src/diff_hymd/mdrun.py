import datetime
import logging

import jax.config as jax_config
import jax.numpy as jnp
import numpy as onp
from jax import random

from .barostat import berendsen, c_rescale
from .file_io import OutDataset, store_data, store_static
from .force import (
    get_angle_energy_and_forces,
    get_bond_energy_and_forces,
    get_dihedral_energy_and_forces,
    get_impropers_energy_and_forces,
    get_protein_dipoles,
    redistribute_dipole_forces,
)
from .input_parser import System
from .integrator import integrate_position, integrate_velocity
from .logger import Logger, format_timedelta
from .pm_functions import (
    get_dipole_forces,
    get_elec_energy_potential_and_forces,
    get_field_energy_and_forces,
    get_field_energy_and_forces_npt,
)
from .thermostat import (
    cancel_com_momentum,
    csvr_thermostat,
    generate_initial_velocities,
)


# pyright: reportUnboundVariable=none
def main(args):
    start_time = datetime.datetime.now()
    key = random.PRNGKey(args.seed)

    cdtype = jnp.complex64
    if args.double_precision:
        # NOTE: this is going to be deprecated in future versions of jax
        jax_config.update("jax_enable_x64", True)
        cdtype = jnp.complex128

    # Load data to System class
    system = System.constructor(args)
    config = system.config
    Logger.rank0.info(f"{config}")

    topol = system.topol
    positions = jnp.mod(system.positions, config.box_size)
    velocities = system.velocities

    if config.start_temperature:
        key, subkey = random.split(key)
        velocities = generate_initial_velocities(velocities, subkey, config)
    elif config.cancel_com_momentum:
        velocities = cancel_com_momentum(velocities, config)

    # Initialize forces
    field_forces = jnp.zeros_like(positions)
    bond_forces = jnp.zeros_like(positions)
    angle_forces = jnp.zeros_like(positions)
    dihedral_forces = jnp.zeros_like(positions)
    improper_forces = jnp.zeros_like(positions)
    elec_forces = jnp.zeros_like(positions)
    reconstr_forces = jnp.zeros_like(positions)

    # Initialize energies
    bond_energy = 0.0
    angle_energy = 0.0
    dihedral_energy = 0.0
    field_energy = 0.0
    elec_energy = 0.0

    # Initialize pressure
    bond_pressure, angle_pressure, dihedral_pressure, field_pressure = 0, 0, 0, 0

    # Initialize field stuff
    # fog = force on grid
    field_fog = jnp.zeros((config.n_types, 3, *(config.mesh_size)))
    phi = jnp.zeros((config.n_types, *config.mesh_size))
    chi = jnp.zeros((config.n_types, config.n_types))
    elec_potential = jnp.zeros(config.mesh_size)

    type_mask = {}
    for i, ti in enumerate(config.unique_types):
        type_mask[i] = jnp.where(system.types == ti, 1.0, 0.0).reshape(-1, 1)
        chi = chi.at[i].set(config.chi[config.type_to_chi[ti]])
    chi = chi.reshape(config.n_types, config.n_types, 1, 1, 1)

    if config.coulombtype and system.charges is not None:
        elec_fog = jnp.zeros((3, *config.mesh_size))
        elec_energy, elec_potential, elec_forces = get_elec_energy_potential_and_forces(
            positions, elec_fog, system.charges, config
        )
    if topol.bonds:
        bond_energy, bond_forces, bond_pressure = get_bond_energy_and_forces(
            bond_forces, positions, config.box_size, *topol.bonds_2
        )
    if topol.angles:
        angle_energy, angle_forces, angle_pressure = get_angle_energy_and_forces(
            angle_forces, positions, config.box_size, *topol.bonds_3
        )
    if topol.dihedrals:
        (
            dihedral_energy,
            dihedral_forces,
            _,
            dihedral_pressure,
        ) = get_dihedral_energy_and_forces(
            dihedral_forces, positions, config.box_size, *topol.bonds_4
        )
    if topol.impropers:
        improper_energy, improper_forces = get_impropers_energy_and_forces(
            improper_forces, positions, config.box_size, *topol.bonds_impr
        )
        dihedral_energy += improper_energy

    # Setup dipoles
    # TODO: should apply only if we have proteins
    if topol.dihedrals:
        dip_fog = jnp.zeros((3, *config.mesh_size))
        # CHECK: should be +2? Check for missing dipole at the start of the sequence?
        n_dip = topol.dihedrals + 1
        dip_charges = jnp.hstack((jnp.full(n_dip, 0.25), jnp.full(n_dip, -0.25)))
        dip_charges = dip_charges.reshape((2 * n_dip, 1))

        # Step 0
        transfer_matrices, dip_positions = get_protein_dipoles(
            positions, config.box_size, *topol.bonds_d
        )
        dip_forces = get_dipole_forces(
            dip_positions, dip_charges, dip_fog, n_dip, config
        )
        reconstr_forces = redistribute_dipole_forces(
            reconstr_forces, dip_forces, transfer_matrices, *topol.bonds_d
        )

    if config.pressure or config.barostat:
        phi_fourier = jnp.zeros((config.n_types, *config.fft_shape), dtype=cdtype)
        field_energy, field_forces, field_pressure = get_field_energy_and_forces_npt(
            # fmt: off
            positions, phi, phi_fourier, field_forces, field_fog, chi, type_mask, elec_potential, config
        )
    else:
        field_energy, field_forces = get_field_energy_and_forces(
            positions, phi, field_forces, field_fog, chi, type_mask, config
        )

    kinetic_energy = 0.5 * config.mass * jnp.sum(velocities * velocities)
    out_dataset = OutDataset(
        args.destdir,
        args.output,
        double_out=False,
    )

    store_static(
        out_dataset,
        system.names,
        onp.asarray(system.types),
        system.indices,
        config,
        topol.bonds_2[0],
        topol.bonds_2[1],
        topol.molecules,
        molecules=system.molecules,
        velocity_out=True,
        force_out=True,
        charges=False,
    )

    if config.n_print > 0:
        step = 0
        frame = 0
        temperature = (2 / 3) * kinetic_energy / (config.R * config.n_particles)

        if config.pressure or config.barostat:
            kinetic_pressure = 2.0 / 3.0 * kinetic_energy
            pressure = (
                kinetic_pressure
                + (field_pressure - field_energy - elec_energy)
                + bond_pressure
                + angle_pressure
                + dihedral_pressure
            ) / config.volume
        else:
            pressure = 0.0

        store_data(
            out_dataset,
            step,
            frame,
            system.indices,
            onp.asarray(positions),
            onp.asarray(velocities),
            onp.asarray(field_forces),
            temperature,
            pressure,
            kinetic_energy,
            bond_energy,
            angle_energy,
            dihedral_energy,
            field_energy,
            elec_energy,
            config,
            velocity_out=True,
            force_out=True,
            charge_out=False,
            dump_per_particle=False,
        )

    loop_start_time = datetime.datetime.now()
    last_step_time = datetime.datetime.now()
    ###################
    # # # MD LOOP # # #
    ###################
    for step in range(1, config.n_steps + 1):
        current_step_time = datetime.datetime.now()
        if step == 1 and args.verbose > 1:
            Logger.rank0.log(logging.INFO, f"MD step = {step:10d}")
        else:
            # log_step = False
            # if config.n_steps < 1000:
            #     log_step = True
            if jnp.mod(step, config.n_print) == 0:
                #     log_step = True
                # if log_step:
                step_t = current_step_time - last_step_time
                tot_t = current_step_time - loop_start_time
                # CHECK: should this be (step + 1) or just step?
                ns_sim = (step + 1) * config.outer_ts / 1000

                seconds_per_day = 24 * 60 * 60
                seconds_elapsed = tot_t.days * seconds_per_day
                seconds_elapsed += tot_t.seconds
                seconds_elapsed += 1e-6 * tot_t.microseconds
                minutes_elapsed = seconds_elapsed / 60
                hours_elapsed = minutes_elapsed / 60
                days_elapsed = hours_elapsed / 24

                ns_per_day = ns_sim / days_elapsed
                hours_per_ns = hours_elapsed / ns_sim
                steps_per_s = (step + 1) / seconds_elapsed
                info_str = (
                    f"Time elapsed: {days_elapsed:.0f}-{hours_elapsed%24:02.0f}:{minutes_elapsed%60:02.0f}:{seconds_elapsed%60:02.0f}   Performance: "
                    f"{ns_per_day:.3f} ns/day   {hours_per_ns:.3f} hours/ns   "
                    f"{steps_per_s:.3f} steps/s"
                )
                Logger.rank0.log(logging.INFO, info_str)

        # Initial rRESPA velocity step
        velocities = integrate_velocity(
            velocities,
            (field_forces + elec_forces + reconstr_forces) / config.mass,
            config.outer_ts,
        )

        # Inner rRESPA steps
        for _ in range(config.respa_inner):
            velocities = integrate_velocity(
                velocities,
                (bond_forces + angle_forces + dihedral_forces + improper_forces)
                / config.mass,
                config.inner_ts,
            )
            positions = integrate_position(positions, velocities, config.inner_ts)
            positions = jnp.mod(positions, config.box_size)

            # Update fast forces
            if topol.bonds:
                bond_energy, bond_forces, bond_pressure = get_bond_energy_and_forces(
                    bond_forces, positions, config.box_size, *topol.bonds_2
                )
            if topol.angles:
                (
                    angle_energy,
                    angle_forces,
                    angle_pressure,
                ) = get_angle_energy_and_forces(
                    angle_forces, positions, config.box_size, *topol.bonds_3
                )
            if topol.dihedrals:
                (
                    dihedral_energy,
                    dihedral_forces,
                    _,
                    dihedral_pressure,
                ) = get_dihedral_energy_and_forces(
                    dihedral_forces, positions, config.box_size, *topol.bonds_4
                )
            if topol.impropers:
                improper_energy, improper_forces = get_impropers_energy_and_forces(
                    improper_forces, positions, config.box_size, *topol.bonds_impr
                )
                dihedral_energy += improper_energy

            velocities = integrate_velocity(
                velocities,
                (bond_forces + angle_forces + dihedral_forces + improper_forces)
                / config.mass,
                config.inner_ts,
            )

        # Second rRESPA velocity step
        # Barostat
        if config.barostat and (jnp.mod(step, config.n_b) == 0):
            if config.coulombtype and system.charges is not None:
                (
                    elec_energy,
                    elec_potential,
                    elec_forces,
                ) = get_elec_energy_potential_and_forces(
                    positions, elec_fog, system.charges, config
                )

            # Get field pressure terms
            (
                field_energy,
                field_forces,
                field_pressure,
            ) = get_field_energy_and_forces_npt(
                # fmt: off
                positions, phi, phi_fourier, field_forces, field_fog, chi, type_mask, elec_potential, config
            )

            # Calculate pressure
            kinetic_energy = 0.5 * config.mass * jnp.sum(velocities**2)
            kinetic_pressure = 2.0 / 3.0 * kinetic_energy
            pressure = (
                kinetic_pressure
                + (field_pressure - field_energy - elec_energy)
                + bond_pressure
                + angle_pressure
                + dihedral_pressure
            ) / config.volume

            # Call barostat
            if config.barostat == 1:
                positions, config = berendsen(
                    pressure,
                    positions,
                    config,
                )
            elif config.barostat == 2:
                positions, velocities, config, key = c_rescale(
                    pressure,
                    positions,
                    velocities,
                    config,
                    key,
                )

        # Recompute after barostat
        field_energy, field_forces = get_field_energy_and_forces(
            positions, phi, field_forces, field_fog, chi, type_mask, config
        )
        if config.coulombtype and system.charges is not None:
            (
                elec_energy,
                elec_potential,
                elec_forces,
            ) = get_elec_energy_potential_and_forces(
                positions, elec_fog, system.charges, config
            )

        if topol.dihedrals:
            transfer_matrices, dip_positions = get_protein_dipoles(
                positions, config.box_size, *topol.bonds_d
            )
            dip_forces = get_dipole_forces(
                dip_positions, dip_charges, dip_fog, n_dip, config
            )
            reconstr_forces = redistribute_dipole_forces(
                reconstr_forces, dip_forces, transfer_matrices, *topol.bonds_d
            )

        velocities = integrate_velocity(
            velocities,
            (field_forces + elec_forces + reconstr_forces) / config.mass,
            config.outer_ts,
        )

        # Thermostat
        if config.target_temperature:
            velocities, key = csvr_thermostat(velocities, key, config)

        # Remove total linear momentum
        if config.cancel_com_momentum:
            if jnp.mod(step, config.cancel_com_momentum) == 0:
                velocities = cancel_com_momentum(velocities, config)

        # Print trajectory
        if config.n_print > 0:
            # Use normal numpy for IO
            if onp.mod(step, config.n_print) == 0 and step != 0:
                frame = step // config.n_print

                kinetic_energy = 0.5 * config.mass * jnp.sum(velocities**2)
                temperature = (2 / 3) * kinetic_energy / (config.R * config.n_particles)
                if config.pressure or config.barostat:
                    kinetic_pressure = 2.0 / 3.0 * kinetic_energy
                    pressure = (
                        kinetic_pressure
                        + (field_pressure - field_energy - elec_energy)
                        + bond_pressure
                        + angle_pressure
                        + dihedral_pressure
                    ) / config.volume
                else:
                    pressure = 0.0

                store_data(
                    out_dataset,
                    step,
                    frame,
                    system.indices,
                    onp.asarray(positions),
                    onp.asarray(velocities),
                    onp.asarray(field_forces),
                    temperature,
                    pressure,
                    kinetic_energy,
                    bond_energy,
                    angle_energy,
                    dihedral_energy,
                    field_energy,
                    elec_energy,
                    config,
                    velocity_out=True,
                    force_out=True,
                    charge_out=False,
                    dump_per_particle=False,
                )
                if onp.mod(step, config.n_print * config.n_flush) == 0:
                    out_dataset.flush()
        last_step_time = current_step_time

    # End simulation
    end_time = datetime.datetime.now()
    sim_time = end_time - start_time
    setup_time = loop_start_time - start_time
    loop_time = end_time - loop_start_time
    Logger.rank0.log(
        logging.INFO,
        (
            f"Elapsed time: {format_timedelta(sim_time)}   "
            f"Setup time: {format_timedelta(setup_time)}   "
            f"MD loop time: {format_timedelta(loop_time)}"
        ),
    )

    if config.n_print > 0 and jnp.mod(config.n_steps - 1, config.n_print) != 0:
        field_energy, field_forces = get_field_energy_and_forces(
            positions, phi, field_forces, field_fog, chi, type_mask, config
        )
        if config.coulombtype and system.charges is not None:
            (
                elec_energy,
                elec_potential,
                elec_forces,
            ) = get_elec_energy_potential_and_forces(
                positions, elec_fog, system.charges, config
            )
        kinetic_energy = 0.5 * config.mass * jnp.sum(velocities * velocities)

        frame = (step + 1) // config.n_print
        temperature = (2 / 3) * kinetic_energy / (config.R * config.n_particles)

        if config.pressure or config.barostat:
            kinetic_pressure = 2.0 / 3.0 * kinetic_energy
            pressure = (
                kinetic_pressure
                + (field_pressure - field_energy - elec_energy)
                + bond_pressure
                + angle_pressure
                + dihedral_pressure
            ) / config.volume
        else:
            pressure = 0.0

        store_data(
            out_dataset,
            step,
            frame,
            system.indices,
            onp.asarray(positions),
            onp.asarray(velocities),
            onp.asarray(field_forces),
            temperature,
            pressure,
            kinetic_energy,
            bond_energy,
            angle_energy,
            dihedral_energy,
            field_energy,
            elec_energy,
            config,
            velocity_out=True,
            force_out=True,
            charge_out=False,
            dump_per_particle=False,
        )
    out_dataset.close_file()
