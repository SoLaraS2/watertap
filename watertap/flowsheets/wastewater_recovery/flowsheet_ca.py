from pyomo.environ import ConcreteModel, TransformationFactory, Constraint
from pyomo.environ import units as pyunits
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.scaling import set_scaling_factor
from pyomo.network import Arc

# WaterTAP props 
from watertap.core.wt_database import Database
from watertap.core.zero_order_properties import WaterParameterBlock
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock as NaClParameterBlock

# Units models
from watertap.unit_models.zero_order.ultra_filtration_zo import UltraFiltrationZO
from components.chemical_addition_zo import ChemicalAdditionZO
from components.ozone_aop_zo import OzoneAOPZO
from components.uv_aop_zo import UVAOPZO
from components.reverse_osmosis_0D import ReverseOsmosis0D  

# Translators
from components.translators import add_translator
from idaes.models.unit_models import StateJunction


# ---- helpers to wire units by equality----
def _has(obj, attr):
    return hasattr(obj, attr)

def _phase_aware_mass_var(props_state, comp):
    if _has(props_state, "flow_mass_phase_comp"):
        v = props_state.flow_mass_phase_comp["Liq", comp]
        return v, pyunits.get_units(v)
    if _has(props_state, "flow_mass_comp"):
        v = props_state.flow_mass_comp[comp]
        return v, pyunits.get_units(v)
    raise AttributeError("Props missing mass-flow vars (flow_mass_phase_comp or flow_mass_comp).")

def _tie_states(host_block, src, dst, comps_src, comps_dst):
    # 1) match total flow (prefer volumetric, else H2O mass)
    if _has(src, "flow_vol") and _has(dst, "flow_vol"):
        dst_u = pyunits.get_units(dst.flow_vol)
        host_block.add_component(
            f"eq_flow_{id(src)}_{id(dst)}",
            Constraint(expr=dst.flow_vol == pyunits.convert(src.flow_vol, to_units=dst_u)),
        )
    else:
        for water_key in ("H2O", "H2o", "h2o", "water"):
            if (water_key in comps_src) and (water_key in comps_dst):
                s, _  = _phase_aware_mass_var(src, water_key)
                d, du = _phase_aware_mass_var(dst, water_key)
                host_block.add_component(
                    f"eq_flow_{water_key}_{id(src)}_{id(dst)}",
                    Constraint(expr=d == pyunits.convert(s, to_units=du)),
                )
                break
    # 2) map overlap species
    common = sorted(set(comps_src).intersection(comps_dst))
    for j in common:
        s, _  = _phase_aware_mass_var(src, j)
        d, du = _phase_aware_mass_var(dst, j)
        host_block.add_component(
            f"eq_comp_{j}_{id(src)}_{id(dst)}",
            Constraint(expr=d == pyunits.convert(s, to_units=du)),
        )

def configure_ozone_defaults(m, peroxone=False):

    a = m.fs.aop
    a.contact_time[0].fix(8.0)
    a.chemical_flow_mass[0].fix(1e-4)   # kg/s


    if hasattr(a, "chemical_flow_mass"):
        a.chemical_flow_mass[0].fix(1e-4)  # kg/s 
    if hasattr(a, "concentration_time"):           a.concentration_time[0].fix(5.0)    # mg·min/L (CT)
    if hasattr(a, "mass_transfer_efficiency"):     a.mass_transfer_efficiency[0].fix(0.85)



    # --- chemistry ratios ---
    # If pure ozone
    if not peroxone:
        if hasattr(a, "ozone_toc_ratio"):      a.ozone_toc_ratio[0].fix(1.0)   # mg-O3 / mg-TOC
        if hasattr(a, "oxidant_ozone_ratio"):  a.oxidant_ozone_ratio[0].fix(1.0)  # all oxidant = ozone

    # If peroxone (O3 + H2O2) — optional
    if peroxone:
        if hasattr(a, "ozone_toc_ratio"):      a.ozone_toc_ratio[0].fix(0.8)
        if hasattr(a, "oxidant_ozone_ratio"):  a.oxidant_ozone_ratio[0].fix(0.6)  # 60% ozone, 40% H2O2 equiv


def build_flowsheet(use_ozone: bool = False):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.db = Database()

    # Property packages
    m.fs.props_wtp  = WaterParameterBlock(solute_list=["tds", "tss", "toc", "chlorine"])
    m.fs.props_nacl = NaClParameterBlock()

    # Feed (WTP props)
    m.fs.feed = StateJunction(property_package=m.fs.props_wtp)

    # === UF (ZO) ===
    m.fs.uf = UltraFiltrationZO(property_package=m.fs.props_wtp, database=m.db)
    m.fs.uf.recovery_frac_mass_H2O.fix(0.99)
    m.fs.uf.removal_frac_mass_comp[0, "tss"].fix(0.90)
    m.fs.uf.removal_frac_mass_comp[0, "tds"].fix(1e-3)
    m.fs.uf.energy_electric_flow_vol_inlet.fix(0.05)

    # === Translator WTP->NaCl ===
    m.fs.t_wtp_to_nacl = add_translator(m, "t_wtp_to_nacl", m.fs.props_wtp, m.fs.props_nacl)

    # === RO (0D) on NaCl (no Ports) ===
    m.fs.ro = ReverseOsmosis0D(property_package=m.fs.props_nacl)
    for name in ("area", "membrane_area"):
        if hasattr(m.fs.ro, name) and (not getattr(m.fs.ro, name).fixed):
            getattr(m.fs.ro, name).fix(1000.0)
            break

    # pressures
    try: m.fs.ro.feed_side.properties_in[0].pressure.fix(15 * pyunits.bar)
    except Exception: pass
    try: m.fs.ro.feed_side.properties_out[0].pressure.fix(14 * pyunits.bar)
    except Exception: pass
    for obj in (getattr(m.fs.ro, "permeate_side", None), getattr(m.fs.ro, "permeate_side_state", None)):
        if obj is not None:
            for key in ((0, 1.0), (0,)):
                try:
                    obj[key].pressure.fix(1 * pyunits.bar); break
                except Exception: pass

    for n in ("A", "A_perm", "A_water", "A_comp"):
        if hasattr(m.fs.ro, n):
            try: m.fs.ro.A.fix(4e-12 * pyunits.m/pyunits.Pa/pyunits.s)
            except Exception: pass
    for n in ("B", "B_perm", "B_salt", "B_comp"):
        if hasattr(m.fs.ro, n):
            try: m.fs.ro.B.fix(1e-7 * pyunits.m/pyunits.s)
            except Exception: pass

    for st in (getattr(m.fs.ro, "feed_side", None), getattr(m.fs.ro, "permeate_side", None)):
        if st is not None:
            for attr in ("properties_in", "properties_out"):
                try: getattr(st, attr)[0].temperature.fix(298.15 * pyunits.K)
                except Exception: pass
    for j in m.fs.props_nacl.component_list:
        m.fs.ro.recovery_mass_phase_comp[0, "Liq", j].fix(0.45 if j == "H2O" else 0.01)

    # === Translator NaCl->WTP ===
    m.fs.t_nacl_to_wtp = add_translator(m, "t_nacl_to_wtp", m.fs.props_nacl, m.fs.props_wtp)

    # === AOP (generic handle) ===
    if use_ozone:
        m.fs.aop = OzoneAOPZO(property_package=m.fs.props_wtp, database=m.db)
        # safe defaults (only if vars exist)
        if hasattr(m.fs.aop, "removal_frac_mass_comp"):
            m.fs.aop.removal_frac_mass_comp[0, "toc"].fix(0.3)
        if hasattr(m.fs.aop, "energy_electric_flow_vol_inlet"):
            m.fs.aop.energy_electric_flow_vol_inlet.fix(0.05)
    else:
        m.fs.aop = UVAOPZO(property_package=m.fs.props_wtp, database=m.db)
        if hasattr(m.fs.aop, "oxidant_dose"):
            m.fs.aop.oxidant_dose[0].fix(5.0)  # mg/L H2O2
        if hasattr(m.fs.aop, "removal_frac_mass_comp"):
            m.fs.aop.removal_frac_mass_comp[0, "toc"].fix(0.3)
        if hasattr(m.fs.aop, "uv_reduced_equivalent_dose"):
            m.fs.aop.uv_reduced_equivalent_dose[0].fix(100 * pyunits.mJ/pyunits.cm**2)
        if hasattr(m.fs.aop, "uv_transmittance_in"):
            m.fs.aop.uv_transmittance_in[0].fix(0.90)
        if hasattr(m.fs.aop, "energy_electric_flow_vol_inlet"):
            m.fs.aop.energy_electric_flow_vol_inlet.fix(0.05)

    # === Chlorine (ChemicalAdditionZO) ===
    m.fs.cl2 = ChemicalAdditionZO(property_package=m.fs.props_wtp, database=m.db, process_subtype="chlorine")
    if hasattr(m.fs.cl2, "solution_density"):
        m.fs.cl2.solution_density.fix(1000 * pyunits.kg/pyunits.m**3)
    if hasattr(m.fs.cl2, "ratio_in_solution"):
        m.fs.cl2.ratio_in_solution.fix(0.125)
    if hasattr(m.fs.cl2, "chemical_dosage"):
        m.fs.cl2.chemical_dosage[0].fix(2.0)  # mg/L as Cl2
    m.fs.cl2.chemical_flow_vol[0].fix(1e-6 * pyunits.m**3/pyunits.s)
    m.fs.cl2.electricity[0].fix(0.0)

    # === Arcs (use generic aop) ===
    m.fs.feed_to_uf = Arc(source=m.fs.feed.outlet, destination=m.fs.uf.inlet)
    m.fs.uf_to_t1   = Arc(source=m.fs.uf.treated, destination=m.fs.t_wtp_to_nacl.inlet.inlet)
    # (RO is tied by equality, not arcs)
    m.fs.t2_to_aop  = Arc(source=m.fs.t_nacl_to_wtp.outlet.outlet, destination=m.fs.aop.inlet)
    m.fs.product    = StateJunction(property_package=m.fs.props_wtp)

    TransformationFactory("network.expand_arcs").apply_to(m)

    # ---- Tie RO (no Ports) ----
    _tie_states(
        m.fs,
        src=m.fs.t_wtp_to_nacl.outlet.properties[0],
        dst=m.fs.ro.feed_side.properties_in[0],
        comps_src=list(m.fs.props_nacl.component_list),
        comps_dst=list(m.fs.props_nacl.component_list),
    )
    _tie_states(
        m.fs,
        src=m.fs.ro.mixed_permeate[0],
        dst=m.fs.t_nacl_to_wtp.inlet.properties[0],
        comps_src=list(m.fs.props_nacl.component_list),
        comps_dst=list(m.fs.props_nacl.component_list),
    )

    # ---- Tie Cl2 after whichever AOP ----
    src_props = getattr(m.fs.aop, "properties_treated", None)
    if src_props is not None:
        src_props = src_props[0]
    else:
        # fallbacks: treated_state -> properties
        src_props = getattr(m.fs.aop, "treated_state", None)
        src_props = src_props[0] if src_props is not None else m.fs.aop.properties[0]

    _tie_states(
        m.fs,
        src=src_props,
        dst=m.fs.cl2.properties[0],
        comps_src=list(m.fs.props_wtp.component_list),
        comps_dst=list(m.fs.props_wtp.component_list),
    )
    _tie_states(
        m.fs,
        src=m.fs.cl2.properties[0],
        dst=m.fs.product.properties[0],
        comps_src=list(m.fs.props_wtp.component_list),
        comps_dst=list(m.fs.props_wtp.component_list),
    )

    # Nominal feed
    f = m.fs.feed.properties[0]
    f.flow_mass_comp["H2O"].fix(1000.0)
    f.flow_mass_comp["tds"].fix(2.0)
    f.flow_mass_comp["tss"].fix(0.01)
    f.flow_mass_comp["toc"].fix(0.5)
    f.flow_mass_comp["chlorine"].fix(1e-12)

    # optional scaling
    try:
        set_scaling_factor(m.fs.uf.properties_in[0].flow_mass_comp["tss"], 1e3)
        set_scaling_factor(m.fs.uf.properties_byproduct[0].flow_mass_comp["tss"], 1e3)
    except Exception:
        pass

    return m


if __name__ == "__main__":
    m = build_flowsheet(use_ozone=True)   # swap True/False to pick ozone vs UV
    configure_ozone_defaults(m, peroxone=False)

    from pyomo.core.base.var import Var

    def list_free(prefix, blk, limit=40):
        print(f"\n[free vars] {prefix}")
        k = 0
        for v in blk.component_data_objects(Var, descend_into=True):
            if (not v.fixed) and (v.value is None) and (not v.is_constant()):
                print(" -", v.name)
                k += 1
                if k >= limit:
                    print(" ... (truncated)")
                    break
        if k == 0:
            print(" (none)")
        return k

    list_free("RO", m.fs.ro)
    list_free("AOP", m.fs.aop)
    list_free("Cl2", m.fs.cl2)

    print("DOF:", degrees_of_freedom(m))
    print("Built flowsheet (UF → RO → AOP → Cl₂; RO+Cl₂ wired by constraints).")
