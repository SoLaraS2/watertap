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
from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock

# Unit models
from watertap.unit_models.zero_order.ultra_filtration_zo import UltraFiltrationZO
# import enums explicitly (they're not attributes of GAC)
from watertap.unit_models.gac import (
    GAC,
    FilmTransferCoefficientType,
    SurfaceDiffusionCoefficientType,
)
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
    m.fs.props_wtp  = WaterParameterBlock(solute_list=["tds","tss","toc","chlorine"])
    m.fs.props_nacl = NaClParameterBlock()
    # MCAS needs MW for each solute
    m.fs.props_mcas = MCASParameterBlock(
        solute_list=["tds","tss","toc","chlorine"],
        mw_data={
            "tds": 0.05844*pyunits.kg/pyunits.mol,      # NaCl proxy
            "tss": 0.10000*pyunits.kg/pyunits.mol,      # generic
            "toc": 0.01200*pyunits.kg/pyunits.mol,      # carbon-equivalent proxy
            "chlorine": 0.070906*pyunits.kg/pyunits.mol # Cl2
        },
    )

    # feed + UF
    m.fs.feed = StateJunction(property_package=m.fs.props_wtp)
    m.fs.uf   = UltraFiltrationZO(property_package=m.fs.props_wtp, database=m.db)
    m.fs.uf.recovery_frac_mass_H2O.fix(0.99)
    m.fs.uf.removal_frac_mass_comp[0,"tss"].fix(0.90)
    m.fs.uf.removal_frac_mass_comp[0,"tds"].fix(1e-3)
    m.fs.uf.energy_electric_flow_vol_inlet.fix(0.05)
    m.fs.uf.electricity[0].fix(0.0)

    # WTP<->NaCl translators + RO
    m.fs.t_wtp_to_nacl = add_translator(m, "t_wtp_to_nacl", m.fs.props_wtp,  m.fs.props_nacl)
    m.fs.ro = ReverseOsmosis0D(property_package=m.fs.props_nacl)
    if hasattr(m.fs.ro, "area"): m.fs.ro.area.fix(1000.0)
    try: m.fs.ro.feed_side.properties_in[0].pressure.fix(15*pyunits.bar)
    except: pass
    try: m.fs.ro.feed_side.properties_out[0].pressure.fix(14*pyunits.bar)
    except: pass
    for j in m.fs.props_nacl.component_list:
        m.fs.ro.recovery_mass_phase_comp[0,"Liq",j].fix(0.45 if j=="H2O" else 0.01)
    ro = m.fs.ro
    if hasattr(ro, "A_comp") and (not ro.A_comp[0, "H2O"].fixed):
        ro.A_comp[0, "H2O"].fix(4e-12 * pyunits.m/pyunits.Pa/pyunits.s)
    m.fs.t_nacl_to_wtp = add_translator(m, "t_nacl_to_wtp", m.fs.props_nacl, m.fs.props_wtp)

    # ozone/uv chain
    last_port = m.fs.t_nacl_to_wtp.outlet.outlet
    if use_ozone:
        m.fs.ozone = OzoneAOPZO(property_package=m.fs.props_wtp, database=m.db)
        m.fs.ozone_arc = Arc(source=last_port, destination=m.fs.ozone.inlet)
        last_port = m.fs.ozone.treated
        configure_ozone_defaults(m.fs.ozone)
    if use_uv:
        m.fs.uv_aop = UVAOPZO(property_package=m.fs.props_wtp, database=m.db)
        m.fs.uv_arc = Arc(source=last_port, destination=m.fs.uv_aop.inlet)
        last_port = m.fs.uv_aop.treated

    # WTP -> MCAS -> GAC -> WTP
    m.fs.t_wtp_to_mcas = add_translator(m, "t_wtp_to_mcas", m.fs.props_wtp,  m.fs.props_mcas)
    m.fs.t_mcas_to_wtp = add_translator(m, "t_mcas_to_wtp", m.fs.props_mcas, m.fs.props_wtp)

    m.fs.gac = GAC(
        property_package=m.fs.props_mcas,
        target_species=["toc"],
        film_transfer_coefficient_type=FilmTransferCoefficientType.fixed,
        surface_diffusion_coefficient_type=SurfaceDiffusionCoefficientType.fixed,
        finite_elements_ss_approximation=5,
    )

    # chlorine + product
    m.fs.cl2 = ChemicalAdditionZO(property_package=m.fs.props_wtp, database=m.db, process_subtype="chlorine")
    m.fs.cl2.chemical_dosage[0].fix(2.0)
    m.fs.cl2.chemical_flow_vol[0].fix(1e-6*pyunits.m**3/pyunits.s)
    m.fs.cl2.electricity[0].fix(0.0)
    m.fs.product = StateJunction(property_package=m.fs.props_wtp)

    # arcs
    m.fs.feed_to_uf   = Arc(source=m.fs.feed.outlet,                 destination=m.fs.uf.inlet)
    m.fs.uf_to_trans  = Arc(source=m.fs.uf.treated,                  destination=m.fs.t_wtp_to_nacl.inlet.inlet)
    m.fs.arc_to_mcas  = Arc(source=last_port,                        destination=m.fs.t_wtp_to_mcas.inlet.inlet)
    m.fs.arc_mcas_gac = Arc(source=m.fs.t_wtp_to_mcas.outlet.outlet, destination=m.fs.gac.inlet)
    m.fs.arc_gac_wtp  = Arc(source=m.fs.gac.outlet,                  destination=m.fs.t_mcas_to_wtp.inlet.inlet)
    m.fs.arc_to_cl2   = Arc(source=m.fs.t_mcas_to_wtp.outlet.outlet, destination=m.fs.cl2.inlet)
    m.fs.arc_product  = Arc(source=m.fs.cl2.outlet,                  destination=m.fs.product.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    # tie states (RO translators)
    _tie_states(m.fs, m.fs.t_wtp_to_nacl.outlet.properties[0], m.fs.ro.feed_side.properties_in[0],
                list(m.fs.props_nacl.component_list), list(m.fs.props_nacl.component_list))
    _tie_states(m.fs, m.fs.ro.mixed_permeate[0], m.fs.t_nacl_to_wtp.inlet.properties[0],
                list(m.fs.props_nacl.component_list), list(m.fs.props_nacl.component_list))

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

    # ozone knobs
    if hasattr(m.fs, "ozone"):
        a = m.fs.ozone
        if hasattr(a, "contact_time"):       a.contact_time[0].fix(8.0)
        if hasattr(a, "concentration_time"): a.concentration_time[0].unfix()
        if hasattr(a, "oxidant_dose"):       a.oxidant_dose[0].fix(3.0)
        if hasattr(a, "chemical_flow_mass"): a.chemical_flow_mass[0].unfix()
        if hasattr(a, "mass_transfer_efficiency"): a.mass_transfer_efficiency[0].fix(0.85)
        if hasattr(a, "specific_energy_coeff"): a.specific_energy_coeff[0].fix(0.02)
        if hasattr(a, "ozone_toc_ratio"):      a.ozone_toc_ratio[0].fix(1.0)
        if hasattr(a, "oxidant_ozone_ratio"):  a.oxidant_ozone_ratio[0].fix(1.0)
        if hasattr(a, "concentration_time"):   a.concentration_time[0].fix(5.0)
        if hasattr(a, "chemical_flow_mass"):   a.chemical_flow_mass[0].fix(0.0)

    # uv knobs
    if hasattr(m.fs, "uv_aop"):
        u = m.fs.uv_aop
        if hasattr(u, "uv_reduced_equivalent_dose"): u.uv_reduced_equivalent_dose[0].fix(100 * pyunits.mJ/pyunits.cm**2)
        if hasattr(u, "uv_transmittance_in"):        u.uv_transmittance_in[0].fix(0.90)
        if hasattr(u, "energy_electric_flow_vol_inlet"): u.energy_electric_flow_vol_inlet.fix(0.05)
        if hasattr(u, "oxidant_dose"):              u.oxidant_dose[0].fix(5.0)
        if hasattr(u, "chemical_flow_mass"):        u.chemical_flow_mass[0].fix(0.0)
        if hasattr(u, "electricity"):               u.electricity[0].fix(0.0)

    # GAC knobs / DOF closure
    if hasattr(m.fs, "gac"):
        m.fs.gac.operational_time.fix(1e5 * pyunits.s)
        m.fs.gac.bed_volumes_treated.fix(1000.0)
        g = m.fs.gac
        g.freund_k.fix(10.0)
        g.freund_ninv.fix(0.5)
        g.ds.fix(1e-14 * pyunits.m**2/pyunits.s)
        g.kf.fix(1e-5  * pyunits.m/pyunits.s)
        g.particle_dens_app.fix(1000 * pyunits.kg/pyunits.m**3)
        g.particle_dens_bulk.fix(500  * pyunits.kg/pyunits.m**3)
        g.particle_dia.fix(0.001 * pyunits.m)
        g.velocity_sup.fix(0.001 * pyunits.m/pyunits.s)
        g.ebct.fix(500 * pyunits.s)
        g.conc_ratio_replace.fix(0.10)
        g.a0.fix(1.0); g.a1.fix(1.0)
        g.b0.fix(0.1); g.b1.fix(0.1); g.b2.fix(0.1); g.b3.fix(0.1); g.b4.fix(0.1)

    # CL2: bookkeeping
    if hasattr(m.fs, "cl2"):
        c = m.fs.cl2
        if hasattr(c, "solution_density"):  c.solution_density.fix(1000 * pyunits.kg/pyunits.m**3)
        if hasattr(c, "ratio_in_solution"): c.ratio_in_solution.fix(0.125)

    # debug: list free
    def list_free(prefix, blk, limit=40):
        print(f"\n[free vars] {prefix}")
        k = 0
        for v in blk.component_data_objects(Var, descend_into=True):
            if (not v.fixed) and (v.value is None) and (not v.is_constant()):
                print(" -", v.name); k += 1
                if k >= limit: print(" ... (truncated)"); break
        if k == 0: print(" (none)")
        return k

    list_free("UF", m.fs.uf)
    list_free("RO", m.fs.ro)
    if hasattr(m.fs, "ozone"):  list_free("Ozone", m.fs.ozone)
    if hasattr(m.fs, "uv_aop"): list_free("UV-AOP", m.fs.uv_aop)
    if hasattr(m.fs, "gac"):    list_free("GAC", m.fs.gac)
    list_free("Cl2", m.fs.cl2)

    print("DOF:", degrees_of_freedom(m))

    solver = get_solver()
    results = solver.solve(m, tee=True)

    print(results.solver.termination_condition)
    print("\n=== Stream Table ===")
    def print_stream(label, props):
        print(f"\n--- {label} ---")
        if hasattr(props, "flow_mass_comp"):
            for c in props.flow_mass_comp:
                print(f"{c}: {value(props.flow_mass_comp[c]):.4f} {pyunits.get_units(props.flow_mass_comp[c])}")
        elif hasattr(props, "flow_mass_phase_comp"):
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

    print("\n=== Stream Results ===")
    print_stream("Feed",          m.fs.feed.properties[0])
    print_stream("UF permeate",   m.fs.uf.properties_treated[0])
    print_stream("RO permeate",   m.fs.ro.mixed_permeate[0])
    if hasattr(m.fs, "gac"):
        print_stream("GAC inlet (MCAS)",  m.fs.gac.process_flow.properties_in[0])
        print_stream("GAC outlet (MCAS)", m.fs.gac.process_flow.properties_out[0])
        print_stream("GAC removed (adsorbed)", m.fs.gac.gac_removed[0])
    print_stream("Product",       m.fs.product.properties[0])
    print("DOF:", degrees_of_freedom(m))

