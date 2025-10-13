from pyomo.environ import ConcreteModel, Constraint, value, units as pyunits
from pyomo.core.base.var import Var
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.scaling import set_scaling_factor
from idaes.core.solvers import get_solver
from pyomo.network import Arc
from pyomo.environ import TransformationFactory


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


# ----------------- helpers -----------------
def _has(obj, attr):
    return hasattr(obj, attr)

def _phase_aware_mass_var(props_state, comp):
    if _has(props_state, "flow_mass_phase_comp"):
        v = props_state.flow_mass_phase_comp["Liq", comp]
        return v, pyunits.get_units(v)
    if _has(props_state, "flow_mass_comp"):
        v = props_state.flow_mass_comp[comp]
        return v, pyunits.get_units(v)
    raise AttributeError("Props missing flow_mass vars")

def _tie_states(host_block, src, dst, comps_src, comps_dst):
    if _has(src, "flow_vol") and _has(dst, "flow_vol"):
        dst_u = pyunits.get_units(dst.flow_vol)
        host_block.add_component(
            f"eq_flow_{id(src)}_{id(dst)}",
            Constraint(expr=dst.flow_vol == pyunits.convert(src.flow_vol, to_units=dst_u)),
        )
    else:
        for water_key in ("H2O","h2o","water"):
            if (water_key in comps_src) and (water_key in comps_dst):
                s, _ = _phase_aware_mass_var(src, water_key)
                d, du = _phase_aware_mass_var(dst, water_key)
                host_block.add_component(
                    f"eq_flow_{water_key}_{id(src)}_{id(dst)}",
                    Constraint(expr=d == pyunits.convert(s, to_units=du)),
                )
                break
    common = sorted(set(comps_src).intersection(comps_dst))
    for j in common:
        s, _ = _phase_aware_mass_var(src, j)
        d, du = _phase_aware_mass_var(dst, j)
        host_block.add_component(
            f"eq_comp_{j}_{id(src)}_{id(dst)}",
            Constraint(expr=d == pyunits.convert(s, to_units=du)),
        )
    for attr in ("temperature","pressure"):
        if _has(src, attr) and _has(dst, attr):
            s = getattr(src, attr)
            d = getattr(dst, attr)
            du = pyunits.get_units(d)
            host_block.add_component(
                f"eq_{attr}_{id(src)}_{id(dst)}",
                Constraint(expr=d == pyunits.convert(s, to_units=du)),
            )

def configure_ozone_defaults(ozone_unit):
    a = ozone_unit
    if hasattr(a, "contact_time"): a.contact_time[0].fix(8.0)
    if hasattr(a, "concentration_time"): a.concentration_time[0].unfix()
    if hasattr(a, "oxidant_dose"): a.oxidant_dose[0].fix(3.0)
    if hasattr(a, "chemical_flow_mass"): a.chemical_flow_mass[0].unfix()
    if hasattr(a, "mass_transfer_efficiency"): a.mass_transfer_efficiency[0].fix(0.85)
    if hasattr(a, "specific_energy_coeff"): a.specific_energy_coeff[0].fix(0.02)

def _treated_props(u):
    p = getattr(u, "properties_treated", None)
    if p is not None: return p[0]
    p = getattr(u, "treated_state", None)
    if p is not None: return p[0]
    return u.properties[0]

# ----------------- build flowsheet -----------------
def build_flowsheet(use_ozone=True, use_uv=True):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.db = Database()

    # property packages
    m.fs.props_wtp = WaterParameterBlock(solute_list=["tds","tss","toc","chlorine"])
    m.fs.props_nacl = NaClParameterBlock()

    # feed + UF
    m.fs.feed = StateJunction(property_package=m.fs.props_wtp)
    m.fs.uf = UltraFiltrationZO(property_package=m.fs.props_wtp, database=m.db)
    m.fs.uf.recovery_frac_mass_H2O.fix(0.99)
    m.fs.uf.removal_frac_mass_comp[0,"tss"].fix(0.90)
    m.fs.uf.removal_frac_mass_comp[0,"tds"].fix(1e-3)
    m.fs.uf.energy_electric_flow_vol_inlet.fix(0.05)
    m.fs.uf.electricity[0].fix(0.0)

    # translators + RO
    m.fs.t_wtp_to_nacl = add_translator(m, "t_wtp_to_nacl", m.fs.props_wtp, m.fs.props_nacl)
    m.fs.ro = ReverseOsmosis0D(property_package=m.fs.props_nacl)
    if hasattr(m.fs.ro, "area"): m.fs.ro.area.fix(1000.0)
    try: m.fs.ro.feed_side.properties_in[0].pressure.fix(15*pyunits.bar)
    except: pass
    try: m.fs.ro.feed_side.properties_out[0].pressure.fix(14*pyunits.bar)
    except: pass
    for j in m.fs.props_nacl.component_list:
        m.fs.ro.recovery_mass_phase_comp[0,"Liq",j].fix(0.45 if j=="H2O" else 0.01)
    m.fs.t_nacl_to_wtp = add_translator(m, "t_nacl_to_wtp", m.fs.props_nacl, m.fs.props_wtp)

    # optional ozone / uv
    last_stream = m.fs.t_nacl_to_wtp.outlet.outlet
    if use_ozone:
        m.fs.ozone = OzoneAOPZO(property_package=m.fs.props_wtp, database=m.db)
        m.fs.ozone_arc = Arc(source=last_stream, destination=m.fs.ozone.inlet)
        last_stream = m.fs.ozone.treated
        configure_ozone_defaults(m.fs.ozone)
    if use_uv:
        m.fs.uv_aop = UVAOPZO(property_package=m.fs.props_wtp, database=m.db)
        m.fs.uv_arc = Arc(source=last_stream, destination=m.fs.uv_aop.inlet)
        last_stream = m.fs.uv_aop.treated

    # chlorine + product
    m.fs.cl2 = ChemicalAdditionZO(property_package=m.fs.props_wtp, database=m.db, process_subtype="chlorine")
    m.fs.cl2.chemical_dosage[0].fix(2.0)
    m.fs.cl2.chemical_flow_vol[0].fix(1e-6*pyunits.m**3/pyunits.s)
    m.fs.cl2.electricity[0].fix(0.0)
    m.fs.product = StateJunction(property_package=m.fs.props_wtp)

    # arcs
    m.fs.feed_to_uf = Arc(source=m.fs.feed.outlet, destination=m.fs.uf.inlet)
    m.fs.uf_to_trans = Arc(source=m.fs.uf.treated, destination=m.fs.t_wtp_to_nacl.inlet.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    # tie states (RO, Cl2)
    _tie_states(m.fs, m.fs.t_wtp_to_nacl.outlet.properties[0], m.fs.ro.feed_side.properties_in[0],
                list(m.fs.props_nacl.component_list), list(m.fs.props_nacl.component_list))
    _tie_states(m.fs, m.fs.ro.mixed_permeate[0], m.fs.t_nacl_to_wtp.inlet.properties[0],
                list(m.fs.props_nacl.component_list), list(m.fs.props_nacl.component_list))

    cl2_src = _treated_props(m.fs.uv_aop) if use_uv else ( _treated_props(m.fs.ozone) if use_ozone else m.fs.t_nacl_to_wtp.outlet.properties[0])
    _tie_states(m.fs, cl2_src, m.fs.cl2.properties[0],
                list(m.fs.props_wtp.component_list), list(m.fs.props_wtp.component_list))
    _tie_states(m.fs, m.fs.cl2.properties[0], m.fs.product.properties[0],
                list(m.fs.props_wtp.component_list), list(m.fs.props_wtp.component_list))

    # nominal feed
    f = m.fs.feed.properties[0]
    f.flow_mass_comp["H2O"].fix(1000.0)
    f.flow_mass_comp["tds"].fix(2.0)
    f.flow_mass_comp["tss"].fix(0.01)
    f.flow_mass_comp["toc"].fix(0.5)
    f.flow_mass_comp["chlorine"].fix(1e-12)

    try:
        set_scaling_factor(m.fs.uf.properties_in[0].flow_mass_comp["tss"], 1e3)
    except: pass

    return m

# ----------------- run + print -----------------
if __name__ == "__main__":
    m = build_flowsheet(use_ozone=True, use_uv=True)
    print("DOF:", degrees_of_freedom(m))

    solver = get_solver()
    results = solver.solve(m, tee=True)

    print(results.solver.termination_condition)
    print("\n=== Stream Table ===")
    def print_stream(label, props):
        print(f"\n--- {label} ---")
        if hasattr(props, "flow_mass_comp"):
            # WaterParameterBlock style
            for c in props.flow_mass_comp:
                print(f"{c}: {value(props.flow_mass_comp[c]):.4f} {pyunits.get_units(props.flow_mass_comp[c])}")
        elif hasattr(props, "flow_mass_phase_comp"):
            # NaClParameterBlock style
            for (p, c) in props.flow_mass_phase_comp:
                if str(p) == "Liq":
                    v = props.flow_mass_phase_comp[p, c]
                    print(f"{c}: {value(v):.4e} {pyunits.get_units(v)}")
        if hasattr(props, "flow_vol"):
            print(f"Volumetric flow: {value(props.flow_vol):.4f} {pyunits.get_units(props.flow_vol)}")
        if hasattr(props, "pressure"):
            print(f"Pressure: {value(props.pressure):.4f} {pyunits.get_units(props.pressure)}")
        if hasattr(props, "temperature"):
            print(f"Temperature: {value(props.temperature):.2f} {pyunits.get_units(props.temperature)}")

    # after solver.solve(...)
    print("\n=== Stream Results ===")
    print_stream("Feed", m.fs.feed.properties[0])
    print_stream("UF permeate", m.fs.uf.properties_treated[0])
    print_stream("RO permeate", m.fs.ro.mixed_permeate[0])
    print_stream("Product", m.fs.product.properties[0])


