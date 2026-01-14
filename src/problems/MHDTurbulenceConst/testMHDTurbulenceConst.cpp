//==============================================================================
// Dimensionless MHD turbulence problem with isothermal EOS and runtime ICs.
//==============================================================================

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "QuokkaSimulation.hpp"
#include "fundamental_constants.H"
#include "hydro/EOS.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "util/BC.hpp"

#include <vector>

struct MHDTurbulenceConst {
};

namespace {
struct MHDTurbParams {
	amrex::Real rho0{1.0};
	amrex::Real scalar0{1.0};
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> B0{AMREX_D_DECL(0.0, 0.0, 0.0)};
};

MHDTurbParams g_params;

void read_mhdturb_params()
{
	amrex::ParmParse pp("mhdturb");
	pp.query("rho0", g_params.rho0);
	pp.query("scalar0", g_params.scalar0);

	std::vector<amrex::Real> bvec;
	if (pp.queryarr("B0", bvec) != 0) {
		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(bvec.size()) == AMREX_SPACEDIM, "mhdturb.B0 must have 3 components in 3D");
		for (int d = 0; d < AMREX_SPACEDIM; ++d) {
			g_params.B0[d] = bvec[d];
		}
	}
}
} // namespace

template <> struct quokka::EOS_Traits<MHDTurbulenceConst> {
	static constexpr double gamma = 1.0; // isothermal
	static constexpr double cs_isothermal = 0.1; // dimensionless sound speed
	static constexpr double mean_molecular_weight = C::m_u;
};

template <> struct Physics_Traits<MHDTurbulenceConst> {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = 0;
	static constexpr int numPassiveScalars = numMassScalars + 1;
	static constexpr bool is_self_gravity_enabled = false;
	static constexpr bool is_radiation_enabled = false;
	static constexpr bool is_dust_enabled = false;
	static constexpr int nDustGroups = 1;
	static constexpr bool is_mhd_enabled = true;
	static constexpr int nGroups = 1;
	static constexpr UnitSystem unit_system = UnitSystem::CONSTANTS;
	static constexpr double boltzmann_constant = C::k_B;
	static constexpr double gravitational_constant = 1.0; // dimensionless G for completeness
};

template <> struct HydroSystem_Traits<MHDTurbulenceConst> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct SimulationData<MHDTurbulenceConst> {
};

template <> void QuokkaSimulation<MHDTurbulenceConst>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const MHDTurbParams params = g_params; // capture runtime params

	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const amrex::Real B2 = 0.5 * (params.B0[0] * params.B0[0] + params.B0[1] * params.B0[1] + params.B0[2] * params.B0[2]);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		state_cc(i, j, k, HydroSystem<MHDTurbulenceConst>::density_index) = params.rho0;
		state_cc(i, j, k, HydroSystem<MHDTurbulenceConst>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDTurbulenceConst>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDTurbulenceConst>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDTurbulenceConst>::internalEnergy_index) = 0.0;
		state_cc(i, j, k, HydroSystem<MHDTurbulenceConst>::energy_index) = B2;
		state_cc(i, j, k, HydroSystem<MHDTurbulenceConst>::scalar0_index) = params.scalar0;
	});
}

template <> void QuokkaSimulation<MHDTurbulenceConst>::setInitialConditionsOnGridFaceVars(quokka::grid const &grid_elem)
{
	const MHDTurbParams params = g_params;

	const amrex::Array4<double> &state_fc = grid_elem.array_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const quokka::direction dir = grid_elem.dir_;

	const int bcomp = Physics_Indices<MHDTurbulenceConst>::mhdFirstIndex;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		if (dir == quokka::direction::x) {
			state_fc(i, j, k, bcomp) = params.B0[0];
		} else if (dir == quokka::direction::y) {
			state_fc(i, j, k, bcomp) = params.B0[1];
		} else {
			state_fc(i, j, k, bcomp) = params.B0[2];
		}
	});
}

auto problem_main() -> int
{
	read_mhdturb_params();

	auto BCs_cc = quokka::BC<MHDTurbulenceConst>(quokka::BCType::int_dir, quokka::BCType::int_dir, quokka::BCType::int_dir);

	const int nvars_fc = Physics_Indices<MHDTurbulenceConst>::nvarTotal_fc;
	amrex::Vector<amrex::BCRec> BCs_fc(nvars_fc);
	for (int icomp = 0; icomp < nvars_fc; ++icomp) {
		for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
			BCs_fc[icomp].setLo(idim, amrex::BCType::int_dir);
			BCs_fc[icomp].setHi(idim, amrex::BCType::int_dir);
		}
	}

	QuokkaSimulation<MHDTurbulenceConst> sim(BCs_cc, BCs_fc);
	sim.setInitialConditions();
	sim.evolve();
	return 0;
}
