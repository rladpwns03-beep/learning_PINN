"""Physics operators for D-BJH PINN."""

from d_bjh_pinn.physics.kelvin import r_k, r_k_derivative
from d_bjh_pinn.physics.mapping import r_p
from d_bjh_pinn.physics.tcurve import t_curve
from d_bjh_pinn.physics.volterra import build_volterra_operator

__all__ = ["r_k", "r_k_derivative", "r_p", "t_curve", "build_volterra_operator"]
