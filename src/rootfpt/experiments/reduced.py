"""Reduced equal-budget stochastic root simulator."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from rootfpt.carbon import CarbonLedger
from rootfpt.development import LateralDevelopment, LateralStatus
from rootfpt.environment import ResourcePatch, TransientPatchField, scenario_patches
from rootfpt.geometry.coverage import RectDomain, graph_tube_metrics
from rootfpt.hydraulics import (
    HydraulicSegment,
    maturation_multiplier,
    solve_hydraulic_network,
)
from rootfpt.random import RandomStreamManager
from rootfpt.sensors import build_sensor
from rootfpt.tips import (
    RootSegment,
    TipState,
    TipStatus,
    angular_difference,
    step_orientation,
    straight_step,
)


@dataclass
class ReducedSimulationResult:
    metrics: dict[str, Any]
    segments: list[RootSegment]
    warnings: list[str]
    seed_manifest: dict[str, Any]


def _jitter_patches(
    patches: tuple[ResourcePatch, ...],
    rng: np.random.Generator,
    position_sd: float,
    amplitude_sd: float,
    lifetime_scale: float,
) -> tuple[ResourcePatch, ...]:
    output = []
    for patch in patches:
        centre = tuple(np.asarray(patch.centre) + rng.normal(0.0, position_sd, size=2))
        amplitude = max(0.05, patch.amplitude + float(rng.normal(0.0, amplitude_sd)))
        output.append(
            replace(
                patch,
                centre=centre,
                amplitude=amplitude,
                lifetime=patch.lifetime * lifetime_scale,
                decay_time=patch.decay_time * lifetime_scale,
            )
        )
    return tuple(output)


def _path_to_collar(
    segment_id: int | None,
    segments_by_id: dict[int, RootSegment],
    resistance_scale: float,
) -> tuple[bool, float, float, float]:
    path_length = 0.0
    raw_resistance = 0.0
    seen: set[int] = set()
    current = segment_id
    while current is not None:
        if current in seen or current not in segments_by_id:
            return False, path_length, math.inf, 0.0
        seen.add(current)
        segment = segments_by_id[current]
        path_length += segment.length
        raw_resistance += segment.length / max(segment.radius**4, 1e-16)
        current = segment.parent_segment_id
    resistance = resistance_scale * raw_resistance
    conductance = 1.0 / (1.0 + resistance)
    return True, path_length, resistance, conductance


def _hydraulic_path_flow(
    *,
    segment_id: int,
    segments_by_id: dict[int, RootSegment],
    contact_resource: float,
    time: float,
    root_radius: float,
    axial_conductivity: float,
    radial_conductance: float,
    collar_potential: float,
    soil_potential_scale: float,
) -> tuple[float, float]:
    """Solve the collar-contact path, equivalent when only that path has uptake."""
    path: list[RootSegment] = []
    current: int | None = segment_id
    seen: set[int] = set()
    while current is not None:
        if current in seen or current not in segments_by_id:
            return 0.0, math.inf
        seen.add(current)
        segment = segments_by_id[current]
        path.append(segment)
        current = segment.parent_segment_id
    path.reverse()
    hydraulic_edges = []
    path_length = 0.0
    for index, segment in enumerate(path):
        path_length += segment.length
        age_factor = maturation_multiplier(max(0.0, time - segment.created_time), 2.0)
        radius_factor = (segment.radius / root_radius) ** 4
        is_contact = index == len(path) - 1
        hydraulic_edges.append(
            HydraulicSegment(
                start=index,
                end=index + 1,
                length=segment.length,
                axial_conductivity=axial_conductivity * radius_factor * age_factor,
                radial_conductance=radial_conductance if is_contact else 0.0,
                soil_potential=contact_resource * soil_potential_scale
                if is_contact
                else 0.0,
            )
        )
    solution = solve_hydraulic_network(
        hydraulic_edges,
        collar_node=0,
        collar_potential=collar_potential,
    )
    return max(0.0, solution.collar_delivered_flow), path_length


def _reflect_position(
    position: np.ndarray,
    orientation: float,
    domain: RectDomain,
) -> tuple[np.ndarray, float]:
    x, z = position
    if x < domain.x_min:
        x = 2.0 * domain.x_min - x
        orientation = math.pi - orientation
    elif x > domain.x_max:
        x = 2.0 * domain.x_max - x
        orientation = math.pi - orientation
    if z < domain.z_min:
        z = 2.0 * domain.z_min - z
        orientation = -orientation
    elif z > domain.z_max:
        z = 2.0 * domain.z_max - z
        orientation = -orientation
    return np.array([x, z]), orientation


def simulate_reduced(config: dict[str, Any], replicate_master_seed: int) -> ReducedSimulationResult:
    """Run one reduced replicate and preserve all failures."""
    streams = RandomStreamManager(replicate_master_seed)
    environment_rng = streams.generator("environment")
    development_rng = streams.generator("development")
    sensor_rng = streams.generator("sensor")

    domain_cfg = config["domain"]
    time_cfg = config["time"]
    root_cfg = config["root"]
    branch_cfg = config["branching"]
    branching_enabled = bool(branch_cfg.get("enabled", True))
    carbon_cfg = config["carbon"]
    environment_cfg = config["environment"]
    sensor_cfg = config["sensor"]
    success_cfg = config["success"]
    hydraulic_cfg = config.get("hydraulics", {})
    hydraulics_enabled = bool(hydraulic_cfg.get("enabled", False))
    numerics_cfg = config["numerics"]

    domain = RectDomain(
        float(domain_cfg["x_min"]),
        float(domain_cfg["x_max"]),
        float(domain_cfg["z_min"]),
        float(domain_cfg["z_max"]),
    )
    patches = _jitter_patches(
        scenario_patches(str(environment_cfg["scenario"])),
        environment_rng,
        float(environment_cfg["position_jitter"]),
        float(environment_cfg["amplitude_jitter"]),
        float(environment_cfg.get("lifetime_scale", 1.0)),
    )
    environment = TransientPatchField(
        patches,
        base_moisture=float(environment_cfg["base_moisture"]),
        depth_gradient=float(environment_cfg["depth_gradient"]),
    )
    sensor = build_sensor(
        str(sensor_cfg["policy"]),
        noise=float(sensor_cfg["noise"]),
        memory_time=float(sensor_cfg["memory_time"]),
        delay=float(sensor_cfg["delay"]),
    )
    ledger = CarbonLedger(
        initial_budget=float(carbon_cfg["initial_budget"]),
        length_cost=float(carbon_cfg["length_cost"]),
        branch_initiation_cost=float(carbon_cfg["branch_initiation_cost"]),
        maintenance_cost=float(carbon_cfg["maintenance_cost"]),
        sensing_cost=float(carbon_cfg["sensing_cost"]),
    )
    development = LateralDevelopment(
        mean_spacing=float(branch_cfg["mean_spacing"]),
        spacing_shape=float(branch_cfg["spacing_shape"]),
        mean_emergence_delay=float(branch_cfg["emergence_delay"]),
        abortion_probability=float(branch_cfg["abortion_probability"]),
        dormancy_probability=float(branch_cfg["dormancy_probability"]),
        daughter_angle_mean=float(branch_cfg["daughter_angle_mean"]),
        daughter_angle_sd=float(branch_cfg["daughter_angle_sd"]),
        maximum_order=int(root_cfg["maximum_order"]),
        rng=development_rng,
    )
    initial = TipState(
        tip_id=0,
        parent_tip_id=None,
        parent_segment_id=None,
        root_order=0,
        root_type="primary",
        position=np.asarray(root_cfg["initial_position"], dtype=float),
        orientation=float(root_cfg["initial_orientation"]),
        age=0.0,
        radius=float(root_cfg["radius"]),
        arc_length=0.0,
        status=TipStatus.ACTIVE,
        sensor_memory=np.zeros(2),
        emergence_time=0.0,
        circumnutation_phase=0.0,
        next_lateral_arc=math.inf,
    )
    development.initialize_tip(initial)
    if not branching_enabled:
        # Consume the same initialization draw as the legacy large-spacing
        # approximation, then make the developmental switch exact.
        initial.next_lateral_arc = math.inf
    tips: list[TipState] = [initial]
    segments: list[RootSegment] = []
    segments_by_id: dict[int, RootSegment] = {}
    warnings: list[str] = []
    next_tip_id = 1
    next_segment_id = 0
    branch_count = 0
    time = 0.0
    dt = float(time_cfg["dt"])
    t_max = float(time_cfg["t_max"])
    maximum_steps = int(numerics_cfg["maximum_steps"])
    maximum_tips = int(root_cfg["maximum_tips"])
    mortality_rate = float(root_cfg["mortality_rate"])

    first_geo = math.nan
    first_use = math.nan
    first_hydraulic = math.nan
    first_cumulative_use = math.nan
    failure_cause = "deadline_reached"
    water_contacted = 0.0
    water_delivered = 0.0
    hydraulic_volume = 0.0
    maximum_hydraulic_flow = 0.0
    minimum_resistance = math.inf
    lifetime_at_contact = math.nan

    for _ in range(maximum_steps):
        if time >= t_max - 1e-12:
            break
        active = [tip for tip in tips if tip.status == TipStatus.ACTIVE]
        if not active:
            failure_cause = "extinction_of_active_tips"
            break
        living_length = sum(segment.length for segment in segments)
        if not ledger.charge_maintenance(living_length, dt):
            failure_cause = "carbon_exhaustion"
            for tip in active:
                tip.status = TipStatus.STOPPED
            break
        if not ledger.charge_sensing(len(active), dt):
            failure_cause = "carbon_exhaustion"
            for tip in active:
                tip.status = TipStatus.STOPPED
            break

        requested_distances = []
        for tip in active:
            moisture, _ = environment.value_gradient(tip.position, time)
            speed_factor = 0.45 + 0.55 * moisture
            requested_distances.append(float(root_cfg["elongation_speed"]) * speed_factor * dt)
        total_requested = sum(requested_distances)
        affordable = ledger.affordable_growth(total_requested)
        growth_scale = affordable / total_requested if total_requested > 0 else 0.0
        if affordable <= 1e-15:
            failure_cause = "carbon_exhaustion"
            for tip in active:
                tip.status = TipStatus.STOPPED
            break
        if not ledger.charge_growth(affordable):
            raise RuntimeError("affordable growth charge unexpectedly failed")

        for tip, requested in zip(active, requested_distances, strict=True):
            gravity_target = math.pi / 2.0
            gravity_drift = float(root_cfg["gravitropism"]) * math.sin(
                angular_difference(gravity_target, tip.orientation)
            )
            sensor_drift = sensor.turning_drift(
                tip=tip,
                environment=environment,
                time=time,
                dt=dt,
                gain=float(root_cfg["hydrotropism"]),
                rng=sensor_rng,
            )
            boundary_drift = 0.0
            margin = float(root_cfg["boundary_margin"])
            if tip.position[0] < domain.x_min + margin:
                boundary_drift += float(root_cfg["boundary_avoidance"])
            elif tip.position[0] > domain.x_max - margin:
                boundary_drift -= float(root_cfg["boundary_avoidance"])
            tip.orientation = step_orientation(
                orientation=tip.orientation,
                drift=gravity_drift + sensor_drift + boundary_drift,
                rotational_diffusion=float(root_cfg["rotational_diffusion"]),
                dt=dt,
                rng=development_rng,
            )
            distance = requested * growth_scale
            start = tip.position.copy()
            end = straight_step(start, tip.orientation, distance)
            end, tip.orientation = _reflect_position(end, tip.orientation, domain)
            actual_distance = float(np.linalg.norm(end - start))
            previous_arc = tip.arc_length
            tip.position = end
            tip.arc_length += actual_distance
            tip.age += dt
            segment = RootSegment(
                segment_id=next_segment_id,
                parent_segment_id=tip.parent_segment_id,
                producing_tip_id=tip.tip_id,
                root_order=tip.root_order,
                start=(float(start[0]), float(start[1])),
                end=(float(end[0]), float(end[1])),
                radius=tip.radius,
                created_time=time,
            )
            segments.append(segment)
            segments_by_id[next_segment_id] = segment
            tip.parent_segment_id = next_segment_id
            if branching_enabled:
                development.register_growth(
                    tip=tip,
                    segment_id=next_segment_id,
                    start_position=start,
                    end_position=end,
                    previous_arc_length=previous_arc,
                    time=time,
                )
            next_segment_id += 1

            moisture, _ = environment.value_gradient(tip.position, time)
            patch = environment.patch_diagnostics(tip.position, time)
            if moisture >= float(success_cfg["geometric_threshold"]):
                water_contacted += float(patch["resource"]) * dt
                connected, path_length, resistance, conductance = _path_to_collar(
                    tip.parent_segment_id,
                    segments_by_id,
                    float(success_cfg["resistance_scale"]),
                )
                minimum_resistance = min(minimum_resistance, resistance)
                if math.isnan(first_geo):
                    first_geo = time + dt
                    lifetime_at_contact = float(patch["remaining_lifetime"])
                transport_delay = path_length / float(success_cfg["transport_speed"])
                instantaneous_delivery = conductance * float(patch["resource"])
                if (
                    connected
                    and ledger.remaining > 0
                    and conductance >= float(success_cfg["functional_flow_threshold"])
                    and float(patch["remaining_lifetime"]) > transport_delay
                    and float(patch["resource"]) >= float(success_cfg["resource_threshold"])
                ):
                    water_delivered += instantaneous_delivery * dt
                    if (
                        math.isnan(first_use)
                        and water_delivered
                        >= float(success_cfg["cumulative_resource_threshold"])
                    ):
                        first_use = time + dt

                if (
                    hydraulics_enabled
                    and connected
                    and ledger.remaining > 0
                    and float(patch["resource"])
                    >= float(success_cfg["resource_threshold"])
                ):
                    hydraulic_flow, hydraulic_path_length = _hydraulic_path_flow(
                        segment_id=tip.parent_segment_id,
                        segments_by_id=segments_by_id,
                        contact_resource=float(patch["resource"]),
                        time=time + dt,
                        root_radius=float(root_cfg["radius"]),
                        axial_conductivity=float(
                            hydraulic_cfg["axial_conductivity"]
                        ),
                        radial_conductance=float(
                            hydraulic_cfg["radial_conductance"]
                        ),
                        collar_potential=float(hydraulic_cfg["collar_potential"]),
                        soil_potential_scale=float(
                            hydraulic_cfg["soil_potential_scale"]
                        ),
                    )
                    transport_delay = hydraulic_path_length / float(
                        success_cfg["transport_speed"]
                    )
                    available = (
                        float(patch["remaining_lifetime"]) > transport_delay
                    )
                    maximum_hydraulic_flow = max(
                        maximum_hydraulic_flow, hydraulic_flow
                    )
                    if (
                        available
                        and hydraulic_flow
                        >= float(hydraulic_cfg["minimum_flow"])
                    ):
                        if math.isnan(first_hydraulic):
                            first_hydraulic = time + dt
                        hydraulic_volume += hydraulic_flow * dt
                        construction_maintenance = (
                            ledger.construction_spent
                            + ledger.branching_spent
                            + ledger.maintenance_spent
                        )
                        net_benefit = (
                            float(hydraulic_cfg["benefit_per_volume"])
                            * hydraulic_volume
                            - construction_maintenance
                        )
                        if (
                            math.isnan(first_cumulative_use)
                            and hydraulic_volume
                            >= float(hydraulic_cfg["minimum_volume"])
                            and net_benefit > 0
                        ):
                            first_cumulative_use = time + dt

            if development_rng.random() < -math.expm1(-mortality_rate * dt):
                tip.status = TipStatus.DEAD

        for site in development.due_sites(time + dt):
            active_count = sum(tip.status == TipStatus.ACTIVE for tip in tips)
            if active_count >= maximum_tips:
                site.status = LateralStatus.DORMANT
                warnings.append(f"maximum_tips reached at lateral site {site.site_id}")
                continue
            if not ledger.charge_branch():
                site.status = LateralStatus.DORMANT
                warnings.append(f"insufficient branch carbon at lateral site {site.site_id}")
                continue
            child = TipState(
                tip_id=next_tip_id,
                parent_tip_id=site.parent_tip_id,
                parent_segment_id=site.parent_segment_id,
                root_order=site.root_order,
                root_type=f"lateral_order_{site.root_order}",
                position=site.position.copy(),
                orientation=development.daughter_orientation(site),
                age=0.0,
                radius=float(root_cfg["radius"])
                * float(root_cfg["radius_order_scale"]) ** site.root_order,
                arc_length=0.0,
                status=TipStatus.ACTIVE,
                sensor_memory=np.zeros(2),
                emergence_time=time + dt,
                circumnutation_phase=0.0,
                next_lateral_arc=math.inf,
            )
            development.initialize_tip(child)
            tips.append(child)
            next_tip_id += 1
            branch_count += 1
            site.status = LateralStatus.EMERGED

        if ledger.closure_error > float(numerics_cfg["tolerance"]):
            failure_cause = "numerical_failure"
            warnings.append(f"carbon closure residual {ledger.closure_error:g}")
            break
        time += dt
    else:
        if time < t_max - 1e-12:
            failure_cause = "numerical_failure"
            warnings.append("maximum_steps reached before t_max")

    if not math.isnan(first_use):
        failure_cause = "none"
    elif ledger.remaining <= 1e-12 and failure_cause == "deadline_reached":
        failure_cause = "carbon_exhaustion"

    segment_array = np.asarray([[segment.start, segment.end] for segment in segments])
    coverage = graph_tube_metrics(
        segments=segment_array,
        radius=float(success_cfg["search_tube_radius"]),
        domain=domain,
        resolution=int(numerics_cfg["coverage_resolution"]),
    )
    positions = np.asarray([tip.position for tip in tips])
    branch_orders = {
        str(order): sum(tip.root_order == order for tip in tips)
        for order in sorted({tip.root_order for tip in tips})
    }
    total_length = sum(segment.length for segment in segments)
    metrics: dict[str, Any] = {
        "replicate_master_seed": replicate_master_seed,
        "T_geo": first_geo,
        "T_use": first_use,
        "T_hydraulic": first_hydraulic,
        "T_cumulative_use": first_cumulative_use,
        "event_observed_geo": int(not math.isnan(first_geo)),
        "event_observed_use": int(not math.isnan(first_use)),
        "event_observed_hydraulic": int(not math.isnan(first_hydraulic)),
        "event_observed_cumulative_use": int(
            not math.isnan(first_cumulative_use)
        ),
        "censored": int(math.isnan(first_use)),
        "failure_time": time,
        "failure_cause": failure_cause,
        "total_root_length": total_length,
        "living_length": total_length,
        "number_of_tips": len(tips),
        "number_of_active_tips": sum(tip.status == TipStatus.ACTIVE for tip in tips),
        "number_of_branches": branch_count,
        "number_of_lateral_sites": len(development.sites),
        "branching_enabled": int(branching_enabled),
        "branch_order_distribution": json.dumps(branch_orders, sort_keys=True),
        "maximum_depth": float(positions[:, 1].max()) if len(positions) else 0.0,
        "lateral_spread": (
            float(positions[:, 0].max() - positions[:, 0].min()) if len(positions) else 0.0
        ),
        "carbon_spent": ledger.total_spent,
        "carbon_remaining": ledger.remaining,
        "carbon_construction": ledger.construction_spent,
        "carbon_branching": ledger.branching_spent,
        "carbon_maintenance": ledger.maintenance_spent,
        "carbon_sensing": ledger.sensing_spent,
        "carbon_closure_residual": ledger.closure_error,
        "water_contacted": water_contacted,
        "water_delivered": water_delivered,
        "hydraulic_delivered_volume": hydraulic_volume,
        "maximum_hydraulic_flow": maximum_hydraulic_flow,
        "hydraulic_water_per_carbon": hydraulic_volume / ledger.total_spent
        if ledger.total_spent > 0
        else 0.0,
        "hydraulics_enabled": int(hydraulics_enabled),
        "water_per_carbon": water_delivered / ledger.total_spent
        if ledger.total_spent > 0
        else 0.0,
        "effective_collar_to_target_resistance": minimum_resistance,
        "target_patch_lifetime_at_contact": lifetime_at_contact,
        "establishment_probability": int(not math.isnan(first_use)),
        "deep_water_depletion": math.nan,
        "oracle_sensor": int(sensor.is_oracle),
        "failed": int(failure_cause == "numerical_failure"),
        **coverage,
    }
    return ReducedSimulationResult(metrics, segments, warnings, streams.manifest())
