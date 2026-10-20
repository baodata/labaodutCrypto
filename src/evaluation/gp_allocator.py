"""Temporary-cost GP reference and constrained receding-horizon approximations.

Bellman: V(p,f)=max_x[-.5(x-p)'L(x-p)
  + d*(x'Bf-g/2*x'Sx+E[V(x,Ff+e)])]. All quantities are per session.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


@dataclass
class GPReference:
    covariance: np.ndarray
    cost: np.ndarray
    transition: np.ndarray
    loading: np.ndarray
    risk_aversion: float
    discount: float
    P: np.ndarray
    Q: np.ndarray
    M: np.ndarray
    D: np.ndarray
    hold_gain: np.ndarray
    signal_gain: np.ndarray
    diagnostics: dict

    def proposal(self, previous, factors):
        previous, factors = np.asarray(previous, float), np.asarray(factors, float)
        if (previous.shape != (len(self.P),) or factors.shape != (self.Q.shape[1],)
                or not np.isfinite(previous).all() or not np.isfinite(factors).all()):
            raise ValueError('Invalid GP state.')
        return self.hold_gain @ previous+self.signal_gain @ factors

    def aim(self, factors):
        if np.linalg.cond(self.P) > 1e12:
            raise RuntimeError('GP aim is ill-conditioned; use direct proposal, never clip aim.')
        return solve(self.P, self.Q @ factors, assume_a='pos')


def solve_gp_reference(covariance, cost, transition, loading, *, risk_aversion,
                       discount, tolerance=1e-10, max_iterations=10000):
    S, L, F, B = [np.asarray(x, dtype=np.float64) for x in (covariance, cost, transition, loading)]
    n = len(S)
    if (S.shape != (n, n) or L.shape != S.shape or B.ndim != 2 or B.shape[0] != n
            or F.shape != (B.shape[1], B.shape[1]) or n < 1
            or not all(np.isfinite(x).all() for x in (S, L, F, B))
            or not np.isfinite([risk_aversion, discount, tolerance]).all()
            or risk_aversion <= 0 or not 0 < discount < 1 or tolerance <= 0
            or type(max_iterations) is not int or max_iterations < 1
            or not np.allclose(S, S.T, atol=1e-12) or not np.allclose(L, L.T, atol=1e-12)
            or min(np.linalg.eigvalsh(S)[0], np.linalg.eigvalsh(L)[0]) <= 0
            or np.max(np.abs(np.linalg.eigvals(F))) > 1+1e-10):
        raise ValueError('GP reference needs finite compatible SPD risk/cost and stable factors.')
    P, Q = np.zeros_like(S), np.zeros_like(B)
    error = np.inf
    for iteration in range(1, max_iterations+1):
        M = L+discount*(risk_aversion*S+P)
        D = discount*(B+Q @ F)
        P_new = L-L @ solve(M, L, assume_a='pos')
        P_new = (P_new+P_new.T)/2
        Q_new = L @ solve(M, D, assume_a='pos')
        error = max(np.linalg.norm(P_new-P)/max(np.linalg.norm(P_new), 1e-14),
                    np.linalg.norm(Q_new-Q)/max(np.linalg.norm(Q_new), 1e-14))
        P, Q = P_new, Q_new
        if error <= tolerance:
            break
    else:
        raise RuntimeError(f'GP Bellman iteration did not converge: residual={error}.')
    M, D = L+discount*(risk_aversion*S+P), discount*(B+Q @ F)
    K, G = solve(M, L, assume_a='pos'), solve(M, D, assume_a='pos')
    residual = max(np.linalg.norm(P-(L-L@K))/max(np.linalg.norm(P), 1e-14),
                   np.linalg.norm(Q-L@G)/max(np.linalg.norm(Q), 1e-14))
    return GPReference(S, L, F, B, risk_aversion, discount, P, Q, M, D, K, G,
        dict(iterations=iteration, relative_update=float(error), bellman_residual=float(residual),
             condition_M=float(np.linalg.cond(M)), min_eigenvalue_P=float(np.linalg.eigvalsh(P)[0]),
             rate_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(np.eye(n)-K)))),
             assumption='stationary_linear_signal_quadratic_temporary_cost_no_constraints'))


def constrained_gp(reference, current, factors, core, budgets, lower, upper,
                   participation_weights, *, linear_cost=0., max_iterations=300,
                   tolerance=1e-9, gap_tolerance=1e-9, desired=None):
    """One-step GP or finite-horizon MPC with unconstrained terminal value.

    Hard equality budgets use a shared deterministic cash ramp, not an
    endogenous market-timing decision. Expected future ADV/NAV/risk are frozen
    at Close. Real fills/costs remain TradingEnv's responsibility.
    desired supplies same-aim speed ablation; horizon must then equal one.
    """
    current, core, factors, budgets, lower, upper, caps = [np.asarray(x, dtype=float)
        for x in (current, core, factors, budgets, lower, upper, participation_weights)]
    H, n = len(budgets), len(core)
    if (H < 1 or current.shape != (n,) or factors.shape != (reference.Q.shape[1],)
            or lower.shape != (H, n) or upper.shape != lower.shape or caps.shape != (n,)
            or np.any(caps < 0) or np.any(lower > upper) or np.any(lower < 0)
            or not all(np.isfinite(x).all() for x in (current, core, factors, budgets, lower, upper, caps))
            or not np.isfinite(linear_cost) or linear_cost < 0
            or np.any(lower.sum(1) > budgets+1e-10) or np.any(upper.sum(1) < budgets-1e-10)
            or desired is not None and (H != 1 or np.shape(desired) != (n,))):
        raise ValueError('Incompatible/infeasible GP constraints/state.')
    count = H*n
    difference = np.eye(count)
    for h in range(1, H):
        difference[h*n:(h+1)*n, (h-1)*n:h*n] = -np.eye(n)
    previous = np.r_[current, np.zeros(count-n)]
    equality = np.kron(np.eye(H), np.ones((1, n)))
    cap = np.tile(caps, H)
    # A feasibility LP provides a valid warm start without silently relaxing caps.
    feasible = linprog(np.zeros(count), A_ub=np.r_[difference, -difference],
        b_ub=np.r_[cap+previous, cap-previous], A_eq=equality, b_eq=budgets,
        bounds=list(zip(lower.ravel(), upper.ravel())), method='highs')
    if not feasible.success:
        raise RuntimeError(f'GP constraints infeasible; no silent fallback: {feasible.message}')
    x0 = feasible.x
    d = reference.discount
    forecast, f = [], factors.copy()
    for h in range(H):
        forecast.append(reference.loading @ f)
        f = reference.transition @ f
    terminal_f = f
    weights = d**np.arange(H)
    L, S, P, Q, gamma = reference.cost, reference.covariance, reference.P, reference.Q, reference.risk_aversion

    def objective(vector):
        w = vector[:count].reshape(H, n)
        active, delta = w-core, (difference @ vector[:count]-previous).reshape(H, n)
        if desired is not None:
            error = w[0]-desired
            value = .5*error @ reference.M @ error
            grad = (reference.M @ error)[None, :]
        else:
            value = 0.
            grad = np.zeros_like(w)
            for h in range(H):
                value += weights[h]*(.5*delta[h]@L@delta[h]
                    +d*(.5*gamma*active[h]@S@active[h]-forecast[h]@active[h]))
                grad[h] += weights[h]*d*(gamma*S@active[h]-forecast[h])
            value += d**H*(.5*active[-1]@P@active[-1]-active[-1]@Q@terminal_f)
            grad[-1] += d**H*(P@active[-1]-Q@terminal_f)
            grad += (difference.T @ np.concatenate([weights[h]*L@delta[h] for h in range(H)])).reshape(H, n)
        if linear_cost:
            value += linear_cost*np.tile(weights[:, None], (1, n)).ravel() @ vector[count:]
            gradient = np.r_[grad.ravel(), linear_cost*np.repeat(weights, n)]
        else:
            gradient = grad.ravel()
        return 1e4*float(value), 1e4*gradient

    constraints = [LinearConstraint(equality if not linear_cost else np.c_[equality, np.zeros((H, count))], budgets, budgets),
        LinearConstraint(difference if not linear_cost else np.c_[difference, np.zeros((count, count))], previous-cap, previous+cap)]
    lo, hi = lower.ravel(), upper.ravel()
    if linear_cost:
        constraints.append(LinearConstraint(np.block([[-difference, np.eye(count)], [difference, np.eye(count)]]),
            np.r_[-previous, previous], np.full(2*count, np.inf)))
        x0 = np.r_[x0, np.abs(difference@x0-previous)]
        lo, hi = np.r_[lo, np.zeros(count)], np.r_[hi, np.ones(count)]
    result = minimize(objective, x0, jac=True, method='SLSQP', bounds=Bounds(lo, hi), constraints=constraints,
        options=dict(maxiter=max_iterations, ftol=tolerance))

    def repair(vector):
        """Repair numerical constraint error, then recertify ORIGINAL objective.

        This is not a substitute portfolio projection: no repaired candidate is
        accepted unless its original convex optimization gap also meets tolerance.
        """
        vector = vector.copy()
        w = vector[:count]
        error = max(float(np.max(abs(equality@w-budgets))),
            float(np.max(abs(difference@w-previous)-cap)),
            float(np.max(lower.ravel()-w)), float(np.max(w-upper.ravel())), 0.)
        distance = 0.
        if error > 1e-10:
            identity = np.eye(count)
            repaired = linprog(np.r_[np.zeros(count), np.ones(count)],
                A_ub=np.block([[difference, np.zeros((count, count))],
                              [-difference, np.zeros((count, count))],
                              [identity, -identity], [-identity, -identity]]),
                b_ub=np.r_[cap+previous, cap-previous, w, -w],
                A_eq=np.c_[equality, np.zeros((H, count))], b_eq=budgets,
                bounds=list(zip(lower.ravel(), upper.ravel()))+[(0., None)]*count, method='highs',
                options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
            if not repaired.success:
                raise RuntimeError(f'Cannot repair numerical QP feasibility: {repaired.message}')
            distance = float(abs(repaired.x[:count]-w).sum())
            vector[:count] = repaired.x[:count]
        if linear_cost:
            vector[count:] = abs(difference@vector[:count]-previous)
        return vector, distance

    def certificate(vector):
        # For convex f, f(x)-f(opt) <= grad f(x)'(x-z), where z minimizes
        # the linearization over the SAME feasible polytope. An LP certificate
        # handles SLSQP status=8 without pretending feasibility proves optimality.
        gradient = objective(vector)[1]/1e4
        eq_a, eq_b, ub_a, ub_b = [], [], [], []
        for constraint in constraints:
            A = np.atleast_2d(constraint.A)
            lb = np.broadcast_to(constraint.lb, (len(A),))
            ub = np.broadcast_to(constraint.ub, (len(A),))
            equal = np.isfinite(lb) & np.isfinite(ub) & (abs(lb-ub) < 1e-12)
            eq_a.extend(A[equal]); eq_b.extend(lb[equal])
            low, high = np.isfinite(lb) & ~equal, np.isfinite(ub) & ~equal
            ub_a.extend(-A[low]); ub_b.extend(-lb[low])
            ub_a.extend(A[high]); ub_b.extend(ub[high])
        oracle = linprog(gradient, A_ub=np.asarray(ub_a) if ub_a else None,
            b_ub=np.asarray(ub_b) if ub_b else None, A_eq=np.asarray(eq_a), b_eq=np.asarray(eq_b),
            bounds=list(zip(lo, hi)), method='highs',
            options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
        return max(0., float(gradient@vector-oracle.fun)) if oracle.success else np.inf

    result.x, repair_distance = repair(result.x)
    gap = certificate(result.x) if np.isfinite(result.x).all() else np.inf
    attempts = 1
    if gap > gap_tolerance:
        attempts = 2
        result = minimize(lambda x: (objective(x)[0]/10, objective(x)[1]/10), result.x,
            jac=True, method='SLSQP', bounds=Bounds(lo, hi), constraints=constraints,
            options=dict(maxiter=max_iterations, ftol=tolerance/10))
        result.x, extra_repair = repair(result.x)
        repair_distance += extra_repair
        gap = certificate(result.x) if np.isfinite(result.x).all() else np.inf
    w = result.x[:count]
    violation = max(float(np.max(np.abs(equality@w-budgets))), float(np.max(np.abs(difference@w-previous)-cap)),
                    float(np.max(lower.ravel()-w)), float(np.max(w-upper.ravel())), 0.)
    if not np.isfinite(result.x).all() or violation > 1e-7 or gap > gap_tolerance:
        raise RuntimeError(f'GP constrained solve failed ({result.message}), primal={violation}, convex gap={gap}.')
    first = w[:n]
    first_delta = first-current
    return first, dict(status='optimal_for_declared_QP', iterations=int(result.nit),
        solver_status=int(result.status), solver_success=bool(result.success),
        certified_by_linearization_LP=True, convex_objective_gap=gap, solver_attempts=attempts,
        numerical_feasibility_repair_l1=repair_distance,
        primal_residual=violation, objective=float(objective(result.x)[0]/1e4), planning_horizon=H,
        planned_quadratic_cost=float(.5*first_delta@L@first_delta),
        planned_linear_cost=float(linear_cost*np.abs(first_delta).sum()),
        active_variance=float((first-core)@S@(first-core)),
        bound_legs=int(np.sum((abs(first-lower[0]) < 1e-7) | (abs(first-upper[0]) < 1e-7))))


def reference_diagnostics():
    """Numerical acceptance artifact: scalar oracle, long horizon and units.

    These calculations are part of every research run, not portfolio alpha
    evidence. A failure stops the runner before processing market outcomes.
    """
    S, L, F, B = np.array([[.04]]), np.array([[.08]]), np.array([[.8]]), np.array([[.01]])
    gamma, d = 3., .99
    ref = solve_gp_reference(S, L, F, B, risk_aversion=gamma, discount=d)
    lam = float(L[0, 0]/S[0, 0])
    q = gamma*d+lam*(1-d)
    a = 2*gamma*lam*d/(np.sqrt(q*q+4*gamma*lam*d*d)+q)
    scalar_error = abs(float((1-ref.hold_gain)[0, 0])-a/lam)
    p, f = np.array([.2]), np.array([.7])
    proposal = ref.proposal(p, f)
    identity_error = float(np.max(abs(proposal-(p+(np.eye(1)-ref.hold_gain)@(ref.aim(f)-p)))))
    foc = float(np.max(abs(ref.M@proposal-L@p-ref.D@f)))
    # Backward induction with zero terminal value, independent fixed budget.
    P, Q = np.zeros((1, 1)), np.zeros((1, 1))
    for _ in range(2000):
        M, D = L+d*(gamma*S+P), d*(B+Q@F)
        P, Q = L-L@np.linalg.solve(M, L), L@np.linalg.solve(M, D)
    finite_error = float(np.max(abs(proposal-np.linalg.solve(L+d*(gamma*S+P), L@p+d*(B+Q@F)@f))))
    from src.env.quadratic_cost import weight_to_holdings_cost
    prices, nav = np.array([100.]), 100000.
    shares_cost = weight_to_holdings_cost(L, prices, nav)
    dw = np.array([.02]); dx = nav*dw/prices
    unit_error = abs(float(.5*dx@shares_cost@dx-nav*.5*dw@L@dw))
    cheap = solve_gp_reference(S, L*1e-8, F, B, risk_aversion=gamma, discount=d)
    cheap_error = float(np.max(abs(cheap.proposal(p, f)-np.linalg.solve(gamma*S, B@f))))
    expensive = solve_gp_reference(S, L*100, F, B, risk_aversion=gamma, discount=d)
    speeds = [float(1-model.hold_gain[0, 0]) for model in (cheap, ref, expensive)]
    if not speeds[0] > speeds[1] > speeds[2]:
        raise RuntimeError('GP scalar cost/speed monotonicity failed.')
    multi = solve_gp_reference(np.array([[.04, .01], [.01, .09]]),
        np.array([[.06, .005], [.005, .03]]), np.diag([.7, .95]),
        np.array([[.01, .002], [-.003, .01]]), risk_aversion=gamma, discount=d)
    pm, fm = np.array([.2, -.1]), np.array([.7, -.4])
    xm = multi.proposal(pm, fm)
    multi_foc = float(np.max(abs(multi.M@xm-multi.cost@pm-multi.D@fm)))
    multi_identity = float(np.max(abs(xm-(pm+(np.eye(2)-multi.hold_gain)@(multi.aim(fm)-pm)))))
    errors = dict(scalar_rate_error=scalar_error, aim_identity_error=identity_error,
        foc_residual=foc, finite_horizon_error=finite_error, money_unit_error=unit_error,
        low_cost_frictionless_error=cheap_error, multi_asset_foc=multi_foc,
        multi_asset_aim_identity=multi_identity)
    if max(errors.values()) > 1e-8:
        raise RuntimeError(f'GP numerical acceptance failed: {errors}')
    return dict(errors, tolerance=1e-8, scalar_speeds_low_medium_high_cost=speeds, reference=ref.diagnostics)
