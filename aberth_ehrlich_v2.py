# Aberth–Ehrlich estable con tolerancia 1e-10


import mpmath as mp

# ============================================================
# STABLE ABERTH-EHRLICH
# ============================================================

# Precisión decimal
mp.mp.dps = 50

# Tolerancia global
TOL = mp.mpf('1e-10')
EPS = mp.mpf('1e-20')


# ============================================================
# NORMALIZACIÓN
# ============================================================
def normalize_coeffs(coeffs):

    coeffs = [mp.mpc(c) for c in coeffs]

    mx = max(abs(c) for c in coeffs)

    return [c / mx for c in coeffs]


# ============================================================
# ESCALAMIENTO
# ============================================================
def scale_polynomial(coeffs):

    n = len(coeffs) - 1

    a0 = coeffs[0]
    an = coeffs[-1]

    if abs(a0) == 0 or abs(an) == 0:
        return coeffs, mp.mpf(1)

    alpha = abs(an / a0) ** (1 / n)

    scaled = [
        coeffs[i] * alpha ** (n - i)
        for i in range(n + 1)
    ]

    return scaled, alpha


# ============================================================
# HORNER
# ============================================================
def horner(coeffs, z):

    result = coeffs[0]

    for c in coeffs[1:]:
        result = result * z + c

    return result


# ============================================================
# DERIVADA
# ============================================================
def horner_derivative(coeffs, z):

    n = len(coeffs) - 1

    result = coeffs[0] * n

    for i in range(1, n):
        result = result * z + coeffs[i] * (n - i)

    return result


# ============================================================
# RADIO DE CAUCHY
# ============================================================
def cauchy_radius(coeffs):

    a0 = abs(coeffs[0])

    mx = max(abs(c) for c in coeffs[1:])

    return 1 + mx / a0


# ============================================================
# INICIALIZACIÓN
# ============================================================
def initial_guesses(n, R):

    roots = []

    for k in range(n):

        theta = 2 * mp.pi * k / n

        perturb = 0.01 * mp.e ** (1j * (k + 1))

        z = (R + perturb) * mp.e ** (1j * theta)

        roots.append(mp.mpc(z))

    return roots


# ============================================================
# ABERTH-EHRLICH
# ============================================================
def aberth_ehrlich(
    coeffs,
    max_iter=5000,
    tol=TOL
):

    coeffs = normalize_coeffs(coeffs)

    n = len(coeffs) - 1

    R = cauchy_radius(coeffs)

    z = initial_guesses(n, R)

    for iteration in range(max_iter):

        max_corr = mp.mpf(0)

        converged = True

        for k in range(n):

            zk = z[k]

            p = horner(coeffs, zk)

            dp = horner_derivative(coeffs, zk)

            if abs(dp) < EPS:
                converged = False
                continue

            frac = p / dp

            summation = mp.mpc(0)

            for j in range(n):

                if j != k:

                    diff = zk - z[j]

                    if abs(diff) < EPS:
                        diff += (
                            mp.mpf('1e-15')
                            * (mp.rand() + 1j * mp.rand())
                        )

                    summation += 1 / diff

            denom = 1 - frac * summation

            if abs(denom) < EPS:
                converged = False
                continue

            correction = frac / denom

            z[k] -= correction

            corr_abs = abs(correction)

            if corr_abs > max_corr:
                max_corr = corr_abs

            if corr_abs > tol:
                converged = False

        if converged and max_corr < tol:

            print(f"Converged in {iteration} iterations")

            return z

    print("WARNING: did not fully converge")

    return z


# ============================================================
# REFINAMIENTO NEWTON
# ============================================================
def refine_roots(coeffs, roots, steps=15):

    for _ in range(steps):

        for i in range(len(roots)):

            z = roots[i]

            p = horner(coeffs, z)

            dp = horner_derivative(coeffs, z)

            if abs(dp) < EPS:
                continue

            roots[i] -= p / dp

    return roots


# ============================================================
# ROOT FINDER
# ============================================================
def find_roots(coeffs):

    coeffs = [mp.mpc(c) for c in coeffs]

    coeffs = normalize_coeffs(coeffs)

    scaled, alpha = scale_polynomial(coeffs)

    roots = aberth_ehrlich(scaled)

    roots = refine_roots(scaled, roots)

    # deshacer escalamiento
    roots = [r * alpha for r in roots]

    return roots


# ============================================================
# ERROR MÁXIMO
# ============================================================
def max_residual(coeffs, roots):

    coeffs = normalize_coeffs(coeffs)

    vals = [abs(horner(coeffs, r)) for r in roots]

    return max(vals)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\nEjemplo: 1 0 -1  => x^2 - 1\n")

    entrada = input("Coefficients: ")

    coeffs = [complex(x) for x in entrada.split()]

    roots = find_roots(coeffs)

    print("\nRoots:\n")

    for r in roots:
        print(mp.nstr(r, 20))

    err = max_residual(coeffs, roots)

    print("\nMaximum residual:")
    print(mp.nstr(err, 10))

    if err < TOL:
        print("\nPrecisión alcanzada: < 1e-10")
    else:
        print("\nPrecisión NO alcanzada")
