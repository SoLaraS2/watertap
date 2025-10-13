"""
Creates a Block with two StateJunctions (from_pkg -> to_pkg) and constraints to
match volumetric flow and component mass flows where names overlap.
"""
from pyomo.environ import Block, Constraint
from pyomo.environ import units as pyunits
from idaes.models.unit_models import StateJunction
from idaes.core.util.initialization import propagate_state


def add_translator(m, name, from_pkg, to_pkg):
    """
    Create a translator block `name` on m.fs that exposes:
      - <name>.inlet  : StateJunction(from_pkg)
      - <name>.outlet : StateJunction(to_pkg)
      - constraints to map flow + species (overlap)
    Returns the created Block.
    """
    blk = Block(concrete=True)
    setattr(m.fs, name, blk)

    blk.inlet  = StateJunction(property_package=from_pkg)
    blk.outlet = StateJunction(property_package=to_pkg)

    # Build maps
    from_comps = list(from_pkg.component_list)
    to_comps   = list(to_pkg.component_list)
    common = sorted(set(from_comps).intersection(to_comps))
    missing_on_to = sorted(set(to_comps) - set(from_comps))

    # ---------- helpers ----------
    def _has(obj, attr):
        return hasattr(obj, attr)

    def _mass_flow_var(props_state, comp):
        """
        Return (var, units) for component mass flow on this props state.
        Prefers phase-indexed ('flow_mass_phase_comp["Liq", comp]') if present;
        otherwise falls back to non-phase ('flow_mass_comp[comp]').
        """
        if _has(props_state, "flow_mass_phase_comp"):
            v = props_state.flow_mass_phase_comp["Liq", comp]
            return v, pyunits.get_units(v)
        if _has(props_state, "flow_mass_comp"):
            v = props_state.flow_mass_comp[comp]
            return v, pyunits.get_units(v)
        raise AttributeError(
            "Property package lacks mass-flow variables "
            "(expected flow_mass_phase_comp or flow_mass_comp)."
        )

    # ---------- match total flow (prefer flow_vol; else H2O mass) ----------
    def _eq_flow_rule(b):
        props_in  = b.inlet.properties[0]
        props_out = b.outlet.properties[0]
        if _has(props_in, "flow_vol") and _has(props_out, "flow_vol"):
            tgt_u = pyunits.get_units(props_out.flow_vol)
            return props_out.flow_vol == pyunits.convert(props_in.flow_vol, to_units=tgt_u)
        # fallback: match H2O mass flow if volumetric not available
        for water_key in ("H2O", "H2o", "h2o", "water"):
            if water_key in from_comps and water_key in to_comps:
                src_v, _ = _mass_flow_var(props_in, water_key)
                tgt_v, tgt_u = _mass_flow_var(props_out, water_key)
                return tgt_v == pyunits.convert(src_v, to_units=tgt_u)
        return Constraint.Skip

    blk.eq_flow = Constraint(rule=_eq_flow_rule)

    # ---------- map common component mass flows (phase-aware) ----------
    blk.eq_comp = Constraint(common)
    for j in common:
        tgt_v, tgt_u = _mass_flow_var(blk.outlet.properties[0], j)
        src_v, _     = _mass_flow_var(blk.inlet.properties[0], j)
        blk.eq_comp[j] = tgt_v == pyunits.convert(src_v, to_units=tgt_u)

    # ---------- zero target-only components (use correct units) ----------
    blk.eq_zero = Constraint(missing_on_to)
    for j in missing_on_to:
        tgt_v, tgt_u = _mass_flow_var(blk.outlet.properties[0], j)
        blk.eq_zero[j] = tgt_v == 0 * tgt_u

    # Minimal initializer
    def initialize():
        blk.inlet.initialize()
        propagate_state(blk.inlet.outlet)
        blk.outlet.initialize()

    blk.initialize = initialize
    return blk
