"""
Creates a Block with two StateJunctions (from_pkg -> to_pkg) and constraints to
match total flow and component flows where names overlap.
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
      - constraints to map flow + species (overlap), mass- or molar-based.
    Returns the created Block.
    """
    blk = Block(concrete=True)
    setattr(m.fs, name, blk)

    blk.inlet  = StateJunction(property_package=from_pkg)
    blk.outlet = StateJunction(property_package=to_pkg)

    from_comps = list(from_pkg.component_list)
    to_comps   = list(to_pkg.component_list)
    common = sorted(set(from_comps).intersection(to_comps))
    missing_on_to = sorted(set(to_comps) - set(from_comps))

    def _has(obj, attr):
        return hasattr(obj, attr)

    # choose a common component-flow basis available on BOTH sides
    def _basis(props_in, props_out):
        if _has(props_in, "flow_mass_phase_comp") and _has(props_out, "flow_mass_phase_comp"):
            return "mass_phase"
        if _has(props_in, "flow_mol_phase_comp") and _has(props_out, "flow_mol_phase_comp"):
            return "mol_phase"
        if _has(props_in, "flow_mass_comp") and _has(props_out, "flow_mass_comp"):
            return "mass"
        if _has(props_in, "flow_mol_comp") and _has(props_out, "flow_mol_comp"):
            return "mol"
        # last resort: try mass on both sides ignoring phase vs non-phase if shapes differ
        if _has(props_in, "flow_mass_phase_comp") and _has(props_out, "flow_mass_comp"):
            return "mass_mixed"
        if _has(props_in, "flow_mass_comp") and _has(props_out, "flow_mass_phase_comp"):
            return "mass_mixed"
        if _has(props_in, "flow_mol_phase_comp") and _has(props_out, "flow_mol_comp"):
            return "mol_mixed"
        if _has(props_in, "flow_mol_comp") and _has(props_out, "flow_mol_phase_comp"):
            return "mol_mixed"
        return None

    def _get_var(props_state, comp, basis):
        if basis == "mass_phase":
            v = props_state.flow_mass_phase_comp["Liq", comp]
        elif basis == "mol_phase":
            v = props_state.flow_mol_phase_comp["Liq", comp]
        elif basis == "mass":
            v = props_state.flow_mass_comp[comp]
        elif basis == "mol":
            v = props_state.flow_mol_comp[comp]
        elif basis == "mass_mixed":
            if _has(props_state, "flow_mass_phase_comp"):
                v = props_state.flow_mass_phase_comp["Liq", comp]
            else:
                v = props_state.flow_mass_comp[comp]
        elif basis == "mol_mixed":
            if _has(props_state, "flow_mol_phase_comp"):
                v = props_state.flow_mol_phase_comp["Liq", comp]
            else:
                v = props_state.flow_mol_comp[comp]
        else:
            raise AttributeError("No compatible component flow variables found.")
        return v, pyunits.get_units(v)

    props_in  = blk.inlet.properties[0]
    props_out = blk.outlet.properties[0]
    basis = _basis(props_in, props_out)

    # ----- total flow: prefer flow_vol if both have it -----
    def _eq_flow_rule(b):
        pi = b.inlet.properties[0]
        po = b.outlet.properties[0]
        if _has(pi, "flow_vol") and _has(po, "flow_vol"):
            tgt_u = pyunits.get_units(po.flow_vol)
            return po.flow_vol == pyunits.convert(pi.flow_vol, to_units=tgt_u)
        # fallback: match H2O (or water) on the chosen basis
        for water_key in ("H2O", "H2o", "h2o", "water"):
            if water_key in from_comps and water_key in to_comps and basis:
                src_v, _   = _get_var(pi, water_key, basis)
                tgt_v, tgt_u = _get_var(po, water_key, basis)
                return tgt_v == pyunits.convert(src_v, to_units=tgt_u)
        return Constraint.Skip
    blk.eq_flow = Constraint(rule=_eq_flow_rule)

    # ----- components on common set -----
    blk.eq_comp = Constraint(common)
    for j in common:
        tgt_v, tgt_u = _get_var(props_out, j, basis)
        src_v, _     = _get_var(props_in,  j, basis)
        blk.eq_comp[j] = tgt_v == pyunits.convert(src_v, to_units=tgt_u)

    # ----- zero any components that exist only on the target package -----
    blk.eq_zero = Constraint(missing_on_to)
    for j in missing_on_to:
        tgt_v, tgt_u = _get_var(props_out, j, basis if basis else "mass")
        blk.eq_zero[j] = tgt_v == 0 * tgt_u

    # Minimal initializer
    def initialize():
        blk.inlet.initialize()
        propagate_state(blk.inlet.outlet)
        blk.outlet.initialize()
    blk.initialize = initialize

    return blk
