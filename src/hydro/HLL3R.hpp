#ifndef HLL3R_HPP_ // NOLINT
#define HLL3R_HPP_

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include <AMReX.H>
#include <AMReX_REAL.H>
#include <algorithm>
#include <limits>

#include "hydro/EOS.hpp"
#include "hydro/HydroState.hpp"
#include "util/ArrayView.hpp"
#include "util/valarray.hpp"

namespace quokka::Riemann
{
// HLL3R solver following Bouchut, Klingenberg & Waagan (2010), hereafter BKW10.
// This is the hydro-only version (3-wave relaxation solver).
//
template <typename problem_t, int N_scalars, int N_mscalars, int fluxdim>
AMREX_FORCE_INLINE AMREX_GPU_DEVICE auto HLL3R(quokka::HydroState<N_scalars, N_mscalars> const &sL, quokka::HydroState<N_scalars, N_mscalars> const &sR,
					       const double gamma, const double du, const double dw) -> quokka::valarray<double, fluxdim>
{
	// ignore unused parameters (used for carbuncle correction in HLLC, not needed here)
	amrex::ignore_unused(du);
	amrex::ignore_unused(dw);

	// BKW10: alpha = (gamma + 1) / 2 for ideal gas
	const double alpha = 0.5 * (gamma + 1.0);
	const double dU = sL.u - sR.u;	// velocity jump
	const double dP = sL.P - sR.P;	// pressure jump

	double c_L = NAN;
	double c_R = NAN;
	double S_L = NAN;
	double S_R = NAN;

	if (gamma != 1.0) {
		// Non-isothermal case

		auto [dedr_L, dedp_L, drdp_L, dpdr_s_L, G_L] = quokka::EOS<problem_t>::ComputeOtherDerivatives(sL.rho, sL.P, sL.massScalar);
		auto [dedr_R, dedp_R, drdp_R, dpdr_s_R, G_R] = quokka::EOS<problem_t>::ComputeOtherDerivatives(sR.rho, sR.P, sR.massScalar);

		// BKW10, Eq.(3.14)(3.34)(3.36)
		// In hydro case, a_q = a^0 = sqrt(dp/drho|_s)
		const double a_L = std::sqrt(std::max(0.0, dpdr_s_L));
		const double a_R = std::sqrt(std::max(0.0, dpdr_s_R));

		// Compute relaxation speeds [BKW10, Eqn. (3.13)]
		c_L = sL.rho * a_L + alpha * sL.rho * (std::max(0.0, dU) + std::max(0.0, -dP) / (sL.rho * a_L + sR.rho * a_R));
		c_R = sR.rho * a_R + alpha * sR.rho * (std::max(0.0, dU) + std::max(0.0, dP) / (sL.rho * a_L + sR.rho * a_R));

		// compute wave speeds [BKW10, Eqn.(3.3)]
		S_L = sL.u - c_L / sL.rho;
		S_R = sR.u + c_R / sR.rho;

	} else {
		// In isothermal case, a = cs
		c_L = sL.rho * sL.cs + alpha * sL.rho * (std::max(0.0, dU) + std::max(0.0, -dP) / (sL.rho * sL.cs + sR.rho * sR.cs));
		c_R = sR.rho * sR.cs + alpha * sR.rho * (std::max(0.0, dU) + std::max(0.0, dP) / (sL.rho * sL.cs + sR.rho * sR.cs));

		S_L = sL.u - c_L / sL.rho;
		S_R = sR.u + c_R / sR.rho;
	}

	// Compute speed and pressure of the 'star' state
	// BKW10, Eqn.(3.5)
	// u* = (c_L * u_L + c_R * u_R + [p]) / (c_L + c_R)
	// p* = (c_R * p_L + c_L * p_R + c_L * c_R * [u]) / (c_L + c_R)
	const double S_star = (c_L * sL.u + c_R * sR.u + dP) / (c_L + c_R);
	const double P_star = (c_R * sL.P + c_L * sR.P + c_L * c_R * dU) / (c_L + c_R);

	// Compute density in star left/right states
	// BKW10, Eqn.(3.16)
	// 1/rho*_L = 1/rho_L + (c_R * (-[u]) + [p]) / (c_L * (c_L + c_R))
	// 1/rho*_R = 1/rho_R - (c_L * [u] + [p]) / (c_R * (c_L + c_R))
	const double inv_rho_star_L = 1.0 / sL.rho + (c_R * (-dU) + dP) / (c_L * (c_L + c_R));
	const double inv_rho_star_R = 1.0 / sR.rho - (c_L * dU + dP) / (c_R * (c_L + c_R));
	// Protect against negative or zero density with a tiny relative floor to avoid excessive diffusion
	constexpr double rho_rel_floor = 1.0e-12;
	const double rho_star_L = 1.0 / std::max(inv_rho_star_L, rho_rel_floor / sL.rho);
	const double rho_star_R = 1.0 / std::max(inv_rho_star_R, rho_rel_floor / sR.rho);

	// Compute specific internal energy in star left/right states for TOTAL ENERGY calculation
	// BKW10, Eqn.(3.2), 3rd Riemann Invariant: e - p^2/(2c^2) = const along contact
	// e*_L = e_L + (p*^2 - p_L^2) / (2 * c_L^2)
	// e*_R = e_R + (p*^2 - p_R^2) / (2 * c_R^2)
	// Note: For total energy, we use the internal energy derived from pressure (consistent with EOS)
	const double eint_from_P_L = quokka::EOS<problem_t>::ComputeEintFromPres(sL.rho, sL.P, sL.massScalar) / sL.rho;
	const double eint_from_P_R = quokka::EOS<problem_t>::ComputeEintFromPres(sR.rho, sR.P, sR.massScalar) / sR.rho;
	// Protect against negative internal energy in high Mach number flows
	const double e_star_L = std::max(0.0, eint_from_P_L + (P_star * P_star - sL.P * sL.P) / (2.0 * c_L * c_L));
	const double e_star_R = std::max(0.0, eint_from_P_R + (P_star * P_star - sR.P * sR.P) / (2.0 * c_R * c_R));

	// Compute total energy in star states
	// In 1D Riemann problem, tangential velocities (v, w) are unchanged across all waves
	// E* = rho* * e* + 0.5 * rho* * (u*^2 + v^2 + w^2)
	const double E_star_L = rho_star_L * e_star_L + 0.5 * rho_star_L * (S_star * S_star + sL.v * sL.v + sL.w * sL.w);
	const double E_star_R = rho_star_R * e_star_R + 0.5 * rho_star_R * (S_star * S_star + sR.v * sR.v + sR.w * sR.w);

	const double Eint_star_L = sL.Eint * (rho_star_L / sL.rho);
	const double Eint_star_R = sR.Eint * (rho_star_R / sR.rho);

	/// Compute fluxes using jump conditions
	// For HLL3R, we compute the star state explicitly and then use the Rankine-Hugoniot
	// jump condition to get the flux:
	// F* = F_L + S_L * (U* - U_L)  for the left star state
	// F* = F_R + S_R * (U* - U_R)  for the right star state

	// N.B.: quokka::valarray is written to allow assigning <= fluxdim
	// components, so this works even if there are more components than
	// enumerated in the initializer list. The remaining components are
	// assigned a default value of zero.

	// Construct conservative state vectors
	quokka::valarray<double, fluxdim> U_L = {sL.rho, sL.rho * sL.u, sL.rho * sL.v, sL.rho * sL.w, sL.E, sL.Eint};
	quokka::valarray<double, fluxdim> U_R = {sR.rho, sR.rho * sR.u, sR.rho * sR.v, sR.rho * sR.w, sR.E, sR.Eint};

	// Star state conservative vectors
	// Tangential velocities (v, w) are unchanged, so: (rho*v)* = rho* * v, (rho*w)* = rho* * w
	quokka::valarray<double, fluxdim> U_star_L = {rho_star_L, rho_star_L * S_star, rho_star_L * sL.v, rho_star_L * sL.w, E_star_L, Eint_star_L};
	quokka::valarray<double, fluxdim> U_star_R = {rho_star_R, rho_star_R * S_star, rho_star_R * sR.v, rho_star_R * sR.w, E_star_R, Eint_star_R};

	// The remaining components are passive scalars, so just copy them from
	// x1LeftState and x1RightState into the (left, right) state vectors U_L and U_R
	// Scalars must also satisfy jump condition: scalar* = scalar * (S - u) / (S - S_star)
	for (int n = 0; n < N_scalars; ++n) {
		const int nstart = fluxdim - N_scalars;
		U_L[nstart + n] = sL.scalar[n];
		U_R[nstart + n] = sR.scalar[n];
		// Star state scalars via Rankine-Hugoniot jump condition
		// U_star_L only uses left state data through S_L wave
		// U_star_R only uses right state data through S_R wave
		U_star_L[nstart + n] = sL.scalar[n] * (S_L - sL.u) / (S_L - S_star);
		U_star_R[nstart + n] = sR.scalar[n] * (S_R - sR.u) / (S_R - S_star);
	}

	// Pressure contribution vector for flux calculation
	const quokka::valarray<double, fluxdim> D_L = {0., 1., 0., 0., sL.u, 0.};
	const quokka::valarray<double, fluxdim> D_R = {0., 1., 0., 0., sR.u, 0.};
	// Note: D_star is not needed in HLL3R since we compute star-state fluxes
	// via Rankine-Hugoniot jump conditions directly from F_L/F_R and U*/U

	// Physical fluxes at left and right states
	// F = [rho*u, rho*u^2 + p, rho*u*v, rho*u*w, (E+p)*u, Eint*u]
	const quokka::valarray<double, fluxdim> F_L = sL.u * U_L + sL.P * D_L;
	const quokka::valarray<double, fluxdim> F_R = sR.u * U_R + sR.P * D_R;

	// Star state fluxes computed via jump conditions:
	// F*_L = F_L + S_L * (U*_L - U_L)
	// F*_R = F_R + S_R * (U*_R - U_R)
	const quokka::valarray<double, fluxdim> F_star_L = F_L + S_L * (U_star_L - U_L);
	const quokka::valarray<double, fluxdim> F_star_R = F_R + S_R * (U_star_R - U_R);

	// Open the Riemann fan and select appropriate flux
	quokka::valarray<double, fluxdim> F{};

	// HLL3R flux selection based on wave speeds
	if (S_L > 0.0) {
		// All waves move to the right, use left flux
		F = F_L;
	} else if ((S_star > 0.0) && (S_L <= 0.0)) {
		// Left wave passed, contact moving right, use left star flux
		F = F_star_L;
	} else if ((S_star <= 0.0) && (S_R >= 0.0)) {
		// Contact passed or stationary, right wave hasn't passed, use right star flux
		F = F_star_R;
	} else {
		// S_R < 0.0: All waves move to the left, use right flux
		F = F_R;
	}

	return F;
}
} // namespace quokka::Riemann

#endif // HLL3R_HPP_
