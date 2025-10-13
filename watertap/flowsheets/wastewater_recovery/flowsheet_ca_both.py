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
    # 3) tie temperature / pressure if present on both
    for attr in ("temperature", "pressure"):
        if _has(src, attr) and _has(dst, attr):
            s = getattr(src, attr)
            d = getattr(dst, attr)
            du = pyunits.get_units(d)
            host_block.add_component(
                f"eq_{attr}_{id(src)}_{id(dst)}",
                Constraint(expr=d == pyunits.convert(s, to_units=du)),
            )


def configure_ozone_defaults(ozone_unit, peroxone=False):
    a = ozone_unit
    # choose ONE knob in each coupled pair
    if hasattr(a, "contact_time"):            a.contact_time[0].fix(8.0)        # minutes
    if hasattr(a, "concentration_time"):      a.concentration_time[0].unfix()   # free alt to avoid over-constraint

    if hasattr(a, "oxidant_dose"):
        a.oxidant_dose[0].fix(3.0)            # mg/L dose
    if hasattr(a, "chemical_flow_mass"):
        a.chemical_flow_mass[0].unfix()       # free the flow alternative

    if hasattr(a, "mass_transfer_efficiency"): a.mass_transfer_efficiency[0].fix(0.85)
    if hasattr(a, "specific_energy_coeff"):    a.specific_energy_coeff[0].unfix()  # often computed; unfix if needed

    # chemistry ratios
    if hasattr(a, "ozone_toc_ratio"):      a.ozone_toc_ratio[0].fix(1.0 if not peroxone else 0.8)
    if hasattr(a, "oxidant_ozone_ratio"):  a.oxidant_ozone_ratio[0].fix(1.0 if not peroxone else 0.6)


# ---------- robust helper to fetch treated-state props from a unit ----------
def _treated_props(u):
    # prefer properties_treated[0] if present (ZO convention)
    p = getattr(u, "properties_treated", None)
    if p is not None:
        return p[0]
    # else try treated_state[0], else the generic properties[0]
    p = getattr(u, "treated_state", None)
    if p is not None:
        return p[0]
    return u.properties[0]

def build_flowsheet(use_ozone: bool = False, use_uv: bool = True):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.db = Database()

    # --- property pkgs ---
    m.fs.props_wtp  = WaterParameterBlock(solute_list=["tds", "tss", "toc", "chlorine"])
    m.fs.props_nacl = NaClParameterBlock()

    # --- feed + UF (WTP) ---
    m.fs.feed = StateJunction(property_package=m.fs.props_wtp)

    m.fs.uf = UltraFiltrationZO(property_package=m.fs.props_wtp, database=m.db)
    m.fs.uf.recovery_frac_mass_H2O.fix(0.99)
    m.fs.uf.removal_frac_mass_comp[0, "tss"].fix(0.90)
    m.fs.uf.removal_frac_mass_comp[0, "tds"].fix(1e-3)
    m.fs.uf.energy_electric_flow_vol_inlet.fix(0.05)
    m.fs.uf.electricity[0].fix(0.0)

    # --- translators & RO (NaCl island) ---
    m.fs.t_wtp_to_nacl = add_translator(m, "t_wtp_to_nacl", m.fs.props_wtp, m.fs.props_nacl)

    m.fs.ro = ReverseOsmosis0D(property_package=m.fs.props_nacl)
    for name in ("area", "membrane_area"):
        if hasattr(m.fs.ro, name) and (not getattr(m.fs.ro, name).fixed):
            getattr(m.fs.ro, name).fix(1000.0); break
    try: m.fs.ro.feed_side.properties_in[0].pressure.fix(15 * pyunits.bar)
    except: pass
    try: m.fs.ro.feed_side.properties_out[0].pressure.fix(14 * pyunits.bar)
    except: pass
    for obj in (getattr(m.fs.ro, "permeate_side", None), getattr(m.fs.ro, "permeate_side_state", None)):
        if obj is not None:
            for key in ((0, 1.0), (0,)):
                try: obj[key].pressure.fix(1 * pyunits.bar); break
                except: pass
    for j in m.fs.props_nacl.component_list:
        m.fs.ro.recovery_mass_phase_comp[0, "Liq", j].fix(0.45 if j == "H2O" else 0.01)
    # pick geometry, free the redundant area var if needed
    ro = m.fs.ro

    # set L/W (example values)
    if hasattr(ro, "length"): ro.length.fix(10.0)     # m
    if hasattr(ro, "width"):  ro.width.fix(0.5)       # m
    # water permeability (A) and salt permeability (B)
    if hasattr(ro, "A_comp") and (not ro.A_comp[0, "H2O"].fixed):
        ro.A_comp[0, "H2O"].fix(4e-12 * pyunits.m/pyunits.Pa/pyunits.s)



    for n in ("area", "membrane_area"):
        if hasattr(ro, n) and getattr(ro, n).fixed:
            getattr(ro, n).unfix()


    m.fs.t_nacl_to_wtp = add_translator(m, "t_nacl_to_wtp", m.fs.props_nacl, m.fs.props_wtp)

    # --- optional AOP chain (WTP domain) ---
    last_stream = None  # will hold the "current" upstream Port for arcs

    # after NaCl->WTP translator, we’re back in WTP domain:
    # start from that translator’s outlet Port
    upstream_port = m.fs.t_nacl_to_wtp.outlet.outlet

    if use_ozone:
        m.fs.ozone = OzoneAOPZO(property_package=m.fs.props_wtp, database=m.db)
        m.fs.t2_to_ozone = Arc(source=upstream_port, destination=m.fs.ozone.inlet)
        last_stream = m.fs.ozone.treated  # Port

    if use_uv:
        m.fs.uv_aop = UVAOPZO(property_package=m.fs.props_wtp, database=m.db)
        # pick correct upstream: ozone.treated if ozone exists, else translator outlet
        upstream_for_uv = (last_stream if last_stream is not None else upstream_port)
        m.fs.ozone_to_uv = Arc(source=upstream_for_uv, destination=m.fs.uv_aop.inlet)
        last_stream = m.fs.uv_aop.treated

    # if neither AOP selected, last_stream remains None → chlorine ties directly from translator outlet
    # --- chlorine (no Ports) + product junction ---
    m.fs.cl2 = ChemicalAdditionZO(property_package=m.fs.props_wtp, database=m.db, process_subtype="chlorine")
    if hasattr(m.fs.cl2, "solution_density"):
        m.fs.cl2.solution_density.fix(1000 * pyunits.kg/pyunits.m**3)
    if hasattr(m.fs.cl2, "ratio_in_solution"):
        m.fs.cl2.ratio_in_solution.fix(0.125)
    if hasattr(m.fs.cl2, "chemical_dosage"):
        m.fs.cl2.chemical_dosage[0].fix(2.0)  # mg/L as Cl2
    m.fs.cl2.chemical_flow_vol[0].fix(1e-6 * pyunits.m**3/pyunits.s)
    m.fs.cl2.electricity[0].fix(0.0)

    m.fs.product = StateJunction(property_package=m.fs.props_wtp)

    # --- ARCs that exist before expand ---
    m.fs.feed_to_uf = Arc(source=m.fs.feed.outlet, destination=m.fs.uf.inlet)
    m.fs.uf_to_t1   = Arc(source=m.fs.uf.treated, destination=m.fs.t_wtp_to_nacl.inlet.inlet)
    # RO is wired by equality (no Ports)
    # From NaCl->WTP translator to AOP(s) we added arcs above
    # No arcs for Cl2/Product (state ties)

    # expand arcs now that all arcs are defined
    TransformationFactory("network.expand_arcs").apply_to(m)

    # --- equality ties (RO, Cl2) ---
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

    # upstream for Cl2: last AOP if any, else translator outlet
    if last_stream is None:
        # no AOPs enabled
        cl2_src_props = m.fs.t_nacl_to_wtp.outlet.properties[0]
    else:
        # take treated properties from the last AOP in chain
        last_unit = m.fs.uv_aop if use_uv else m.fs.ozone
        cl2_src_props = _treated_props(last_unit)

    _tie_states(
        m.fs,
        src=cl2_src_props,
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

    # --- nominal feed ---
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
    except:
        pass

    return m



if __name__ == "__main__":
    # pick combos
    m = build_flowsheet(use_ozone=True, use_uv=True)   # both in series
    # m = build_flowsheet(use_ozone=True, use_uv=False)  # ozone only
    # m = build_flowsheet(use_ozone=False, use_uv=True)  # UV only

    # --- ozone: fix one of each coupled pair, free the alt ---
    if hasattr(m.fs, "ozone"):
        a = m.fs.ozone
        if hasattr(a, "contact_time"):            a.contact_time[0].fix(8.0)      # min
        if hasattr(a, "concentration_time"):      a.concentration_time[0].fix(5.0) # 
        if hasattr(a, "oxidant_dose"):            a.oxidant_dose[0].fix(3.0)      # mg/L
        if hasattr(a, "chemical_flow_mass"):      a.chemical_flow_mass[0].unfix() # free alt
        if hasattr(a, "mass_transfer_efficiency"):a.mass_transfer_efficiency[0].fix(0.85)
        # this one is currently FREE on your print; either fix it or unfix depending on model
        if hasattr(a, "specific_energy_coeff"):   a.specific_energy_coeff[0].fix(0.02)  # kWh/m^3 per mg/L (example)

    # --- UV-AOP: your print shows these as FREE; fix the scalar + time-indexed knobs, free the alt flow ---
    if hasattr(m.fs, "uv_aop"):
        u = m.fs.uv_aop
        if hasattr(u, "uv_reduced_equivalent_dose"): u.uv_reduced_equivalent_dose[0].fix(100 * pyunits.mJ/pyunits.cm**2)
        if hasattr(u, "uv_transmittance_in"):        u.uv_transmittance_in[0].fix(0.90)
        # scalar (no [0] in your print) -> fix directly
        if hasattr(u, "energy_electric_flow_vol_inlet"): u.energy_electric_flow_vol_inlet.fix(0.05)   # kWh/m^3
        # choose dose OR flow
        if hasattr(u, "oxidant_dose"):              u.oxidant_dose[0].fix(5.0)     # mg/L H2O2
        if hasattr(u, "chemical_flow_mass"):        u.chemical_flow_mass[0].unfix()
        # some ZO expose an electricity bookkeeping var; you showed it free → fix it
        if hasattr(u, "electricity"):               u.electricity[0].fix(0.0)
    
    # kill the alternate knob: we’re dose-driven
    if hasattr(m.fs, "ozone"):
        configure_ozone_defaults(m.fs.ozone, peroxone=False)
    if hasattr(m.fs, "ozone") and hasattr(m.fs.ozone, "chemical_flow_mass"):
        m.fs.ozone.chemical_flow_mass[0].fix(0.0 * pyunits.kg/pyunits.s)

    if hasattr(m.fs, "uv_aop") and hasattr(m.fs.uv_aop, "chemical_flow_mass"):
        m.fs.uv_aop.chemical_flow_mass[0].fix(0.0 * pyunits.kg/pyunits.s)

    from pyomo.core.base.var import Var
    def list_free(prefix, blk, limit=40):
        print(f"\n[free vars] {prefix}")
        k = 0
        for v in blk.component_data_objects(Var, descend_into=True):
            if (not v.fixed) and (v.value is None) and (not v.is_constant()):
                print(" -", v.name); k += 1
                if k >= limit: print(" ... (truncated)"); break
        if k == 0: print(" (none)")
        return k

    list_free("RO", m.fs.ro)
    if hasattr(m.fs, "ozone"):  list_free("Ozone", m.fs.ozone)
    if hasattr(m.fs, "uv_aop"): list_free("UV-AOP", m.fs.uv_aop)
    list_free("Cl2", m.fs.cl2)
    list_free("UF", m.fs.uf)

    print("DOF:", degrees_of_freedom(m))



    
