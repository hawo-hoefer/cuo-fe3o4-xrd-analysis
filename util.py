from pymatgen.io.cif import CifParser

AVOGADRO_CONST = 6.02214076e23

def mixture_molar_mass(wfs: dict[str, float], densities: dict[str, float], molar_volumes: dict[str, float]):
    inv_mass = 0.0
    for phase, wf in wfs.items():
        phase_molar_mass = densities[phase] * molar_volumes[phase]
        inv_mass += wf / phase_molar_mass

    return 1 / inv_mass
    

def vf_to_wt(vfs: dict[str, float]) -> dict[str, float]:
    densities = {}
    for phase in vfs.keys():
        s = CifParser(f"./cif/{phase}.cif").parse_structures(primitive=False)[0]
        densities[phase] = float(s.density) * 1000

    sum_rho_i_phi_i = 0.0
    for phase, vf in vfs.items():
        sum_rho_i_phi_i += densities[phase] * vf

    wfs = {}
    for phase, vf in vfs.items():
        wfs[phase] = densities[phase] * vf / sum_rho_i_phi_i

    return wfs

def wt_to_vf(wfs: dict[str, float]) -> dict[str, float]:
    densities = {}
    for phase in wfs.keys():
        s = CifParser(f"./cif/{phase}.cif").parse_structures(primitive=False)[0]
        densities[phase] = float(s.density) * 1000

    sum_rho_i_phi_i = 0.0
    for phase, wf in wfs.items():
        sum_rho_i_phi_i += wf / densities[phase]

    vfs = {}
    for phase, wf in wfs.items():
        vfs[phase] = wf / densities[phase]  / sum_rho_i_phi_i

    return vfs


def wt_to_mole_frac_gsas(wfs: dict[str, float]) -> dict[str, float]:
    molar_volumes = {}
    densities = {}
    for phase in wfs.keys():
        s = CifParser(f"./cif/{phase}.cif").parse_structures(primitive=False)[0]
        formula_units = s.composition.num_atoms / s.composition.reduced_composition.num_atoms
        molar_volumes[phase] = AVOGADRO_CONST * s.volume * (1e-10)**3 
        densities[phase] = float(s.density) * 1000
        
    total_molar_mass = mixture_molar_mass(wfs, densities, molar_volumes)


    mol_fracs = {}
    for phase, wf in wfs.items():
        phase_molar_mass = densities[phase] * molar_volumes[phase]
        mol_fracs[phase] = total_molar_mass / phase_molar_mass * wf
               
    return mol_fracs

