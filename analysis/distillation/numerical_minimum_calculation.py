import argparse
from mpmath import mp, mpf, pi, sqrt, sin, gamma, quad, findroot
from tqdm import tqdm
import sys

def h_density_mp(t, p):
    """
    The probability density function using mpmath.
    h_density_mp(t, p) = constant * (sin t)^(p - 2), for t in [0, pi].
    """
    t = mpf(t)
    p = mpf(p)
    if mpf('0') <= t <= pi:
        term1 = mpf('1') / sqrt(pi)
        term2 = gamma(p / 2) / gamma((p - 1) / 2)
        term3 = (sin(t))**(p - 2)
        return term1 * term2 * term3
    return mpf('0')

def survival_function_single_angle_mp(theta, p):
    """
    The survival function for a single random angle, i.e. P(Theta >= theta).
    """
    theta = mpf(theta)
    p = mpf(p)
    if theta < 0:
        return mpf('1.0')
    if theta > pi:
        return mpf('0.0')
    integral = quad(lambda t: h_density_mp(t, p), [theta, pi])
    return integral

def survival_function_min_angle_mp(theta, n, p):
    """
    The survival function of the minimum angle among n angles: P(min >= theta) = [P(Theta >= theta)]^n.
    """
    theta = mpf(theta)
    n = mpf(n)
    p = mpf(p)
    if theta < 0:
        return mpf('1.0')
    if theta > pi:
        return mpf('0.0')
    return survival_function_single_angle_mp(theta, p) ** n

def expected_minimum_angle_numerical_mp(n, p, dps=50, progress_bar=True):
    """
    Numerically calculates the expected minimum angle among n random vectors in R^p.
    The integration is performed from 0 to pi of survival_function_min_angle_mp(theta, n, p).
    
    E[min(Theta)] = \int_0^pi P(min(Theta) >= t) dt.
    
    If 'progress_bar' is True and the terminal supports it, a progress bar is shown.
    """
    if p < 2:
        raise ValueError("Dimension p must be >= 2")
    if n < 1:
        raise ValueError("Number of prior vectors n must be >= 1")

    mp.dps = dps
    n_mp = mpf(n)
    p_mp = mpf(p)

    # Dummy integration to estimate intervals for progress bar
    try:
        quad(lambda t: survival_function_min_angle_mp(t, n_mp, p_mp), [mpf('0'), pi], maxdegree=1)
        total_intervals = 2  # Very rough guess
    except Exception:
        total_intervals = 200  # Fallback if dummy integration fails

    progress_bar_active = progress_bar and sys.stderr.isatty()

    if progress_bar_active:
        pbar = tqdm(total=total_intervals, desc="Integrating", unit="interval")

    def integrand_with_progress(t):
        if progress_bar_active:
            # We can't easily track sub-interval progress with mpmath.
            # This callback is called many times, we just do minimal updates.
            pbar.update(0)
        return survival_function_min_angle_mp(t, n_mp, p_mp)

    # Manual piecewise integration for a simple progress bar demonstration
    num_segments = total_intervals
    step = pi / num_segments
    expected_value_rad = mpf('0')

    for i in range(num_segments):
        lower = mpf(i) * step
        upper = mpf(i + 1) * step
        try:
            segment_integral = quad(integrand_with_progress, [lower, upper])
            expected_value_rad += segment_integral
            if progress_bar_active:
                pbar.update(1)
        except Exception as e:
            if progress_bar_active:
                pbar.set_postfix({"error": str(e)}, refresh=True)
            raise

    if progress_bar_active:
        pbar.close()

    return expected_value_rad

def expected_minimum_angle_approx_mp(n, p, dps=50):
    """
    Closed-form approximation of the expected minimum angle for large n in R^p:
    
    E[min(Theta)] ≈ (1/(p-1)) * gamma(1/(p-1)) * (n*K)^(-1/(p-1)),
    where K = (1/(4 sqrt(pi))) * Gamma(p/2) / Gamma((p+1)/2).
    
    This formula is valid for p > 2. For p = 2, a simpler approach is used.
    """
    if p < 2:
        raise ValueError("Dimension p must be >= 2")
    if n < 1:
        raise ValueError("Number of prior vectors n must be >= 1")

    mp.dps = dps
    n_mp = mpf(n)
    p_mp = mpf(p)

    if p_mp == 2:
        # For p=2, the distribution of a random angle is uniform in [0, pi].
        # The min of n uniform(0, pi) angles has expected value = pi/(n+1).
        # Or equivalently, if we used a small-angle approximation for large n, 
        # we might say ~ (1 / n). But let's do the exact known formula for uniform.
        return pi / (n_mp + 1)
    elif p_mp > 2:
        K = (mpf('1') / (mpf('4') * sqrt(pi))) * gamma(p_mp / 2) / gamma((p_mp + mpf('1')) / 2)
        term1 = mpf('1') / (p_mp - mpf('1'))
        term2 = (n_mp * K)**(-mpf('1') / (p_mp - mpf('1')))
        term3 = gamma(mpf('1') / (p_mp - mpf('1')))
        return term1 * term2 * term3
    else:
        # Shouldn't happen, but just in case
        return mpf('nan')

# ---------------------------------------------------------------------
# Inversion routines: given a target angle, solve for n
# ---------------------------------------------------------------------

def expected_minimum_angle_approx_invert_mp(theta_min, p, dps=50):
    """
    Inverts the large-n approximation to solve for n given E[min(Theta)] = theta_min in R^p.
    
    For p=2, E[min(Theta)] ~ pi/(n+1). So we solve pi/(n+1) = theta_min => n+1 = pi/theta_min => n = pi/theta_min - 1.
    
    For p>2,
      E[min(Theta)] = (1/(p-1)) * gamma(1/(p-1)) * (n*K)^(-1/(p-1))
      => n = ...
    """
    mp.dps = dps
    p_mp = mpf(p)
    th_mp = mpf(theta_min)

    if th_mp <= 0:
        # If target angle is <= 0, no finite n can produce that as an *average* min angle.
        # Return 0 or negative, or just a special value:
        return mpf('0')

    if p_mp < 2:
        raise ValueError("Dimension p must be >= 2")

    if p_mp == 2:
        # Solve pi/(n+1) = theta_min => n+1 = pi/theta_min => n = pi/theta_min - 1
        if th_mp >= pi:
            # If the target angle is >= pi, the only way E[min(Theta)] can be pi or above
            # is if n=1. But actually for n=1, E[theta]=pi/2. There's no way to get E[min] = pi.
            # We'll just return an indicative value:
            return mpf('0')
        # Return the formula. Could be negative if theta_min>pi, but we handled that above.
        return mp.pi / th_mp - mpf('1')
    else:
        # p>2
        # E = term1 * term2 * term3 = (1/(p-1)) * gamma(1/(p-1)) * (n*K)^(-1/(p-1))
        # Let E = theta_min
        # => theta_min / [ (1/(p-1)) * gamma(1/(p-1)) ] = (n*K)^(-1/(p-1))
        # => (n*K) = [ theta_min / ( (1/(p-1)) * gamma(1/(p-1)) ) ] ^ (-(p-1))
        # => n = (1/K) * X,  X = the bracket term
        K = (mpf('1') / (mpf('4') * sqrt(pi))) * gamma(p_mp / mpf('2')) / gamma((p_mp + mpf('1')) / mpf('2'))
        alpha = mpf('1') / (p_mp - mpf('1'))
        c = (alpha * gamma(alpha))  # = (1/(p-1)) * gamma(1/(p-1))
        
        # If c == 0 or something weird, guard:
        if c <= 0:
            return mpf('nan')

        left_side = th_mp / c  # E / ...
        if left_side <= 0:
            return mpf('0')
        # (n*K)^(-alpha) = left_side
        # n*K = left_side^(-1/alpha)
        n_times_K = left_side**(-mpf('1')/alpha)
        n_val = n_times_K / K
        # Because this is an approximate formula, n_val might be < 1 if theta_min is large.
        return n_val

def find_n_for_expected_min_angle_numerical_mp(theta_min, p, dps=50, tolerance=1e-7, max_iter=100, progress_bar=False):
    """
    Numerically solves for n > 0 such that E[min(Theta)] = theta_min.
    We use a simple bracket / bisection approach calling expected_minimum_angle_numerical_mp(n, p).
    
    WARNING: For large n, the numerical approach can be very slow since
    each evaluation integrates from 0 to pi. Consider using --approximate for large n.
    """
    mp.dps = dps
    th_mp = mpf(theta_min)
    p_mp = mpf(p)

    if th_mp <= 0:
        return mpf('0')
    if th_mp >= pi:
        # The min angle cannot realistically have an expected value >= pi unless n=1 and p=2, 
        # but even then E[min(Theta)]< pi. We'll return n=0 as a sign that it's not possible.
        return mpf('0')

    # Define f(n) = E[min(Theta)] - theta_min
    def f(n):
        val = expected_minimum_angle_numerical_mp(n, p_mp, dps=dps, progress_bar=False)
        return val - th_mp

    # 1) Bracket n. We expect that for n=1, E[min(Theta)] is some angle > 0.
    #    Increase n until E[min(Theta)] < theta_min to find an upper bracket.
    n_low = mpf('1')
    f_low = f(n_low)  # E[min] - theta_min at n=1

    if f_low < 0:
        # That means E[min(Theta)](n=1) < theta_min => we want an n < 1 (?), which is not feasible.
        # We'll just return something < 1 to indicate no solution in n>=1.
        return mpf('1') if abs(f_low) < tolerance else mpf('0')

    # Increase n until we cross zero or we reach some large max
    n_high = mpf('2')
    for _ in range(max_iter):
        f_high = f(n_high)
        if f_high < 0:
            break
        n_high *= mpf('2')
    else:
        # If we never broke out, we didn't find a bracket up to large n
        # We'll just return the final n_high. Possibly the angle_min can't get that small.
        return n_high

    # Now we have f(n_low) >= 0, f(n_high) < 0. Bisection to refine
    for _ in range(max_iter):
        n_mid = (n_low + n_high) / mpf('2')
        f_mid = f(n_mid)
        if f_mid == 0 or (n_high - n_low) < mpf(tolerance):
            return n_mid
        if f_mid > 0:
            n_low = n_mid
        else:
            n_high = n_mid

    return (n_low + n_high) / mpf('2')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate or invert the expected minimum angle in R^p with arbitrary precision.")
    
    # Existing arguments
    parser.add_argument("-p", "--dimension", type=int, required=True,
                        help="Dimension of the space (p >= 2)")
    parser.add_argument("-n", "--num_prior_vectors", type=int,
                        help="Number of prior random vectors (n >= 1). "
                             "If omitted and --theta_min is not used, the script will fail. "
                             "If --theta_min is used, we do not need n.")
    parser.add_argument("-a", "--approximate", action="store_true",
                        help="Use the large-n approximation")
    parser.add_argument("-dps", "--precision", type=int, default=50,
                        help="Decimal places of precision for mpmath (default: 50)")
    parser.add_argument("--no_progress", action="store_true",
                        help="Disable the progress bar")

    # New argument for inversion
    parser.add_argument("--theta_min", type=float, default=None,
                        help="Target angle for the expected minimum angle. "
                             "If provided, the script will solve for n. "
                             "Units are radians by default unless --theta_min_degrees is also set.")
    parser.add_argument("--theta_min_degrees", action="store_true",
                        help="If set, interpret --theta_min as degrees instead of radians.")
    
    args = parser.parse_args()
    dimension = args.dimension
    num_prior = args.num_prior_vectors
    use_approx = args.approximate
    precision_dps = args.precision
    disable_progress = args.no_progress
    theta_min_val = args.theta_min
    theta_min_in_degrees = args.theta_min_degrees

    if dimension < 2:
        print("Error: Dimension p must be >= 2")
        exit(1)

    mp.dps = precision_dps

    # If we are asked to invert (i.e. find n such that E[min angle] = theta_min):
    if theta_min_val is not None:
        # Convert to radians if needed
        theta_min_rad = mpf(theta_min_val)
        if theta_min_in_degrees:
            theta_min_rad = theta_min_rad * pi / mpf('180')
        
        if use_approx:
            # Use approximate formula for inversion
            try:
                n_solution = expected_minimum_angle_approx_invert_mp(theta_min_rad, dimension, dps=precision_dps)
                print(f"Inverted (approx) for p={dimension}, E[min(angle)]={theta_min_val}{' deg' if theta_min_in_degrees else ' rad'} => n={n_solution}")
            except ValueError as e:
                print(f"Error in approximate inversion: {e}")
        else:
            # Use numerical approach for inversion
            try:
                n_solution = find_n_for_expected_min_angle_numerical_mp(theta_min_rad,
                                                                        dimension,
                                                                        dps=precision_dps,
                                                                        progress_bar=not disable_progress)
                print(f"Inverted (numerical) for p={dimension}, E[min(angle)]={theta_min_val}{' deg' if theta_min_in_degrees else ' rad'} => n={n_solution}")
            except ValueError as e:
                print(f"Error in numerical inversion: {e}")
            except Exception as e:
                print(f"An error occurred during numerical inversion: {e}")

    else:
        # The original behavior: we must have n provided
        if num_prior is None:
            print("Error: You must provide either --num_prior_vectors or --theta_min.")
            exit(1)

        if num_prior < 1:
            print("Error: Number of prior vectors n must be >= 1")
            exit(1)

        if use_approx:
            # Approximate expected minimum angle
            try:
                expected_angle_rad = expected_minimum_angle_approx_mp(num_prior, dimension, dps=precision_dps)
                expected_angle_deg = expected_angle_rad * (mpf('180') / pi)
                print(f"Approximate expected minimum angle (n={num_prior}, p={dimension}, dps={precision_dps}): {expected_angle_deg} degrees")
            except ValueError as e:
                print(f"Error: {e}")
        else:
            # Numerical expected minimum angle
            try:
                expected_angle_rad = expected_minimum_angle_numerical_mp(num_prior, dimension,
                                                                         dps=precision_dps,
                                                                         progress_bar=not disable_progress)
                expected_angle_deg = expected_angle_rad * (mpf('180') / pi)
                print(f"\nNumerically calculated expected minimum angle (n={num_prior}, p={dimension}, dps={precision_dps}): {expected_angle_deg} degrees")
            except ValueError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"\nAn error occurred during numerical integration: {e}")

