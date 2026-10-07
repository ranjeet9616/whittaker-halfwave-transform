import sympy as sp
import mpmath as mp

print("=== SYMBOLIC ===")
x, lam, a_, b_ = sp.symbols('x lambda alpha beta', positive=True)
g = sp.Function('g')
p = x**(2*a_+1)*sp.exp(-x); w = x**(2*a_)*sp.exp(-x)
q = x**(2*a_-1)*sp.exp(-x)*(a_+sp.Rational(1,4)+(2*a_+4*b_+1)/2*x)
tau = lambda y: (-sp.diff(p*sp.diff(y, x), x) + q*y)/w
y = sp.sqrt(x)*g(x)
lhs = sp.simplify(tau(y)/sp.sqrt(x))
rhs = -x*sp.diff(g(x), x, 2) - (2*a_+2-x)*sp.diff(g(x), x) + (a_+1+2*b_)*g(x)
print("tau(sqrt(x) g)/sqrt(x) - Laguerre form =", sp.simplify(lhs-rhs))

# sqrt(x) cancellation in Wronskian
M_, U_ = sp.Function('M')(x), sp.Function('U')(x)
phi = sp.sqrt(x)*M_; chi = sp.sqrt(x)*U_
print("phi chi' - phi' chi - x(MU'-M'U) =",
      sp.simplify(phi*sp.diff(chi,x)-sp.diff(phi,x)*chi - x*(M_*sp.diff(U_,x)-sp.diff(M_,x)*U_)))

# operational property test function f = sqrt(x) e^{-x/2}
f = sp.sqrt(x)*sp.exp(-x/2)
print("tau f =", sp.simplify(tau(f)))

# worked example closed form: int x^g e^{-px} L_n^g dx
gam, s = sp.Rational(8,5), sp.Rational(1,3)
pp = 1+s
for n in range(4):
    val = sp.integrate(x**gam*sp.exp(-pp*x)*sp.assoc_laguerre(n, gam, x), (x, 0, sp.oo))
    cf = sp.gamma(n+gam+1)/sp.factorial(n)*s**n/pp**(n+gam+1)
    print(f"example n={n}: integral - closed form =", sp.N(val-cf, 20))

print("\n=== NUMERIC ===")
mp.mp.dps = 30
al, be = mp.mpf('0.3'), mp.mpf('0.25')
b = 2*al+2
lam0 = al+1+2*be
A = lambda L: al+1+2*be-L
phiF = lambda X, L: mp.sqrt(X)*mp.hyp1f1(A(L), b, X)
chiF = lambda X, L: mp.sqrt(X)*mp.hyperu(A(L), b, X)
pF = lambda X: X**(2*al+1)*mp.e**(-X)
wF = lambda X: X**(2*al)*mp.e**(-X)

# Wronskian [chi,phi] = p (chi phi' - chi' phi) = Gamma(b)/Gamma(a)
print("Wronskian check:")
for L in [mp.mpf('0.7'), mp.mpf('2.3'), mp.mpc('1.1','0.8')]:
    for X in [mp.mpf('0.5'), mp.mpf(3), mp.mpf(9)]:
        br = pF(X)*(chiF(X,L)*mp.diff(lambda t: phiF(t,L), X) - mp.diff(lambda t: chiF(t,L), X)*phiF(X,L))
        print(f"  L={mp.nstr(L,4)}, x={X}: [chi,phi]={mp.nstr(br,15)}, Gamma(b)/Gamma(a)={mp.nstr(mp.gamma(b)/mp.gamma(A(L)),15)}")

# chi(lambda_n) = (-1)^n (b)_n phi(lambda_n)
print("chi_n / phi_n  vs (-1)^n (b)_n:")
for n in range(4):
    L = lam0+n
    print(f"  n={n}:", mp.nstr(chiF(mp.mpf(2.7),L)/phiF(mp.mpf(2.7),L),15), mp.nstr((-1)**n*mp.rf(b,n),15))

# norms: ||phi_n||^2 = n! Gamma(b)^2/Gamma(b+n)
print("||phi_n||^2 check:")
for n in range(4):
    # use the exact integer parameter a=-n (floating lam0+n would make M grow like e^x)
    nm = mp.quad(lambda X: X*mp.hyp1f1(-n, b, X)**2*wF(X), [0,1,10,40,mp.inf])
    print(f"  n={n}:", mp.nstr(nm,15), mp.nstr(mp.factorial(n)*mp.gamma(b)**2/mp.gamma(b+n),15))

# m-function M(lambda) = Gamma(1-b)Gamma(a)/(Gamma(b)Gamma(a-b+1)) (2alpha not integer)
Mfun = lambda L: mp.gamma(1-b)*mp.gamma(A(L))/(mp.gamma(b)*mp.gamma(A(L)-b+1))
thetaF = lambda X, L: X**(mp.mpf(3)/2-b)*mp.hyp1f1(A(L)-b+1, 2-b, X)/(b-1)
print("theta + M phi = Gamma(a)/Gamma(b) chi ; [theta,phi]=1:")
for L in [mp.mpf('0.7'), mp.mpc('1.1','0.8')]:
    X = mp.mpf('1.7')
    lhs_ = thetaF(X,L)+Mfun(L)*phiF(X,L); rhs_ = mp.gamma(A(L))/mp.gamma(b)*chiF(X,L)
    br = pF(X)*(thetaF(X,L)*mp.diff(lambda t: phiF(t,L), X) - mp.diff(lambda t: thetaF(t,L), X)*phiF(X,L))
    print(f"  L={mp.nstr(L,4)}: diff={mp.nstr(abs(lhs_-rhs_),5)}, [theta,phi]={mp.nstr(br,15)}")
print("Residues of M at lambda_n:")
for n in range(4):
    L = lam0+n; eps = mp.mpf('1e-15')
    r = eps*Mfun(L+eps)
    print(f"  n={n}: eps*M = {mp.nstr(r,12)},  -1/||phi_n||^2 = {mp.nstr(-mp.gamma(b+n)/(mp.factorial(n)*mp.gamma(b)**2),12)}")
print("Im M(lambda+i eps) sign sample (Herglotz test):",
      [mp.nstr(mp.im(Mfun(mp.mpc(t,'0.3'))),5) for t in [-5,0,1.5,3,10]])

# Green's function vs spectral expansion for test function f = sqrt(x) e^{-s x}
mp.mp.dps = 20
s_ = mp.mpf(1)/3; gam = b-1
fT = lambda X: mp.sqrt(X)*mp.e**(-s_*X)
Lr = mp.mpf('0.9')  # not an eigenvalue (lam0 = 1.8)
Wr = mp.gamma(b)/mp.gamma(A(Lr))
def Rf(X):
    I1 = mp.quad(lambda t: phiF(t,Lr)*fT(t)*wF(t), [0, X])
    I2 = mp.quad(lambda t: chiF(t,Lr)*fT(t)*wF(t), [X, X+10, mp.inf])
    return (chiF(X,Lr)*I1 + phiF(X,Lr)*I2)/Wr
def Rf_series(X, N=60):
    tot = 0
    for n in range(N):
        Ln = lam0+n
        yn = mp.sqrt(X)*mp.laguerre(n, gam, X)
        cn = mp.gamma(n+gam+1)/mp.factorial(n)*s_**n/(1+s_)**(n+gam+1)
        nn = mp.gamma(n+gam+1)/mp.factorial(n)
        tot += cn*yn/(nn*(Ln-Lr))
    return tot
print("Resolvent: Green-function integral vs spectral series, lambda=0.9:")
for X in [mp.mpf('0.4'), mp.mpf(2), mp.mpf(6)]:
    print(f"  x={X}:", mp.nstr(Rf(X),14), mp.nstr(Rf_series(X),14))
# also check (tau - lambda) Rf = f numerically at x=2 via ODE form
X0 = mp.mpf(2)
yv = Rf(X0); d1 = mp.diff(Rf, X0, 1); d2 = mp.diff(Rf, X0, 2)
lhs_ode = -(X0*d2 + (2*al+1-X0)*d1) + ((al+mp.mpf(1)/4)/X0 + al+mp.mpf(1)/2+2*be)*yv - Lr*yv
print("  (tau-lambda)(Rf)(2) =", mp.nstr(lhs_ode,12), " f(2) =", mp.nstr(fT(X0),12))

# Hilbert-Schmidt: ||R||_HS^2 = sum 1/(lambda_n-lambda)^2 vs double integral
mp.mp.dps = 15
def cum_phi(X):
    return mp.quad(lambda t: phiF(t,Lr)**2*wF(t), [0, X])
HS = 2*mp.quad(lambda X: chiF(X,Lr)**2*wF(X)*cum_phi(X), [0, 1, 5, 15, 40])/Wr**2
HS_series = mp.nsum(lambda n: 1/(lam0+n-Lr)**2, [0, mp.inf])
print("HS norm^2: double integral =", mp.nstr(HS,10), " sum 1/(lam_n-lam)^2 =", mp.nstr(HS_series,10))

# Transform round trip and Parseval with f = 1/(1+x)  (Whittaker-kernel normalisation)
mp.mp.dps = 20
fR = lambda X: 1/(1+X)
def K(X, n):   # x^{-a-1/2} e^{x/2} W_{kappa_n, mu}(x) = (-1)^n n! sqrt(x) L_n^{(2a+1)}(x)
    return (-1)**n*mp.factorial(n)*mp.sqrt(X)*mp.laguerre(n, gam, X)
# check K formula against whitw
kap = al+1+2; Xc = mp.mpf('1.3')
print("K via whitw vs Laguerre form (n=2):",
      mp.nstr(Xc**(-al-0.5)*mp.e**(Xc/2)*mp.whitw(kap, al+0.5, Xc),15), mp.nstr(K(Xc,2),15))
N_ = lambda n: mp.factorial(n)*mp.gamma(n+2*al+2)
Nmax = 80
F = [mp.quad(lambda X: fR(X)*K(X,n)*wF(X), [0,1,5,20,60,mp.inf]) for n in range(Nmax)]
normf = mp.quad(lambda X: fR(X)**2*wF(X), [0,1,10,mp.inf])
for NN in (10, 20, 40, 80):
    ps = sum(F[n]**2/N_(n) for n in range(NN))
    print(f"Parseval N={NN}: sum={mp.nstr(ps,15)}  ||f||^2={mp.nstr(normf,15)}  rel.gap={mp.nstr((normf-ps)/normf,4)}")
for X in [mp.mpf('0.5'), mp.mpf(2), mp.mpf(5)]:
    rec = sum(F[n]*K(X,n)/N_(n) for n in range(Nmax))
    print(f"Inversion x={X}: f={mp.nstr(fR(X),12)} reconstructed(N=80)={mp.nstr(rec,12)}")

# operational property: W[tau f](u_n) = u_n^2 W[f](u_n), f = sqrt(x)e^{-x/2}
tf = sp.lambdify(x, sp.simplify(tau(f)).subs({a_: sp.Rational(3,10), b_: sp.Rational(1,4)}), 'mpmath')
ff = lambda X: mp.sqrt(X)*mp.e**(-X/2)
for n in range(4):
    lhs_ = mp.quad(lambda X: tf(X)*K(X,n)*wF(X), [0,1,10,mp.inf])
    rhs_ = (lam0+n)*mp.quad(lambda X: ff(X)*K(X,n)*wF(X), [0,1,10,mp.inf])
    print(f"Operational n={n}: W[tau f]={mp.nstr(lhs_,14)}  u_n^2 W[f]={mp.nstr(rhs_,14)}")

# Hankel identification: g_n(r)=r^nu e^{-r^2/2} L_n^nu(r^2), int g_n(r) J_nu(k r) r dr = (-1)^n g_n(k)
nu = gam
for n in range(4):
    gn = lambda r: r**nu*mp.e**(-r**2/2)*mp.laguerre(n, nu, r**2)
    k = mp.mpf('1.4')
    H = mp.quad(lambda r: gn(r)*mp.besselj(nu, k*r)*r, [0, 2, 5, 10, 15])
    print(f"Hankel n={n}: H_nu g_n(k)={mp.nstr(H,14)}  (-1)^n g_n(k)={mp.nstr((-1)**n*gn(k),14)}")
