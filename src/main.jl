# BEEM138 Macroeconomic Theory I - Problem Set 4
# Neoclassical growth model: shooting algorithm
# Run from the project folder:  julia --project=. src/main.jl

using Plots
gr()

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
const A = 1.0
const α = 0.3
const σ = 2.0
const ρ = 0.05
const δ = 0.05
const β = 1 / (1 + ρ)
const T = 60

f(k)  = A * k^α
fp(k) = α * A * k^(α - 1)

# ---------------------------------------------------------------
# (a) Steady state (analytical)
#     β(f'(k*) + 1 - δ) = 1  =>  α A k*^(α-1) = ρ + δ
#     c* = f(k*) - δ k*
# ---------------------------------------------------------------
kstar = (α * A / (ρ + δ))^(1 / (1 - α))
cstar = f(kstar) - δ * kstar
println("(a) Steady state:  k* = $(round(kstar, digits=5)),  c* = $(round(cstar, digits=5))")

# ---------------------------------------------------------------
# (b) Shooting algorithm
# ---------------------------------------------------------------
# Runs the difference equations forward from (k0, c0) with for loops.
# Returns :high (c0 too high), :low (c0 too low) or :ok, the paths, and
# the number of periods simulated before stopping.
function simulate(c0, k0)
    up = k0 <= kstar                        # true if capital must rise toward k*
    k = zeros(T + 1)
    c = zeros(T + 1)
    k[1], c[1] = k0, c0
    for t in 1:T
        k[t+1] = f(k[t]) + (1 - δ) * k[t] - c[t]
        if k[t+1] <= 0                      # capital collapses -> c0 too high
            return :high, k, c, t
        end
        c[t+1] = c[t] * (β * (fp(k[t+1]) + 1 - δ))^(1 / σ)
        if up
            if k[t+1] < k[t]                # k falls before reaching k* -> c0 too high
                return :high, k, c, t + 1
            elseif c[t+1] < c[t]            # k overshoots k*, c falls -> c0 too low
                return :low, k, c, t + 1
            end
        else
            if k[t+1] > k[t]                # k rises although k0 > k* -> c0 too low
                return :low, k, c, t + 1
            elseif c[t+1] > c[t]            # k undershoots k*, c rises -> c0 too high
                return :high, k, c, t + 1
            end
        end
    end
    return :ok, k, c, T + 1
end

# Bisection on c0 (step iii: adjust the guess until the path converges)
function shoot(k0; tol = 1e-12)
    lo, hi = 0.0, f(k0) + (1 - δ) * k0      # c0 cannot exceed total resources (k1 >= 0)
    c0 = 0.5 * (lo + hi)
    for _ in 1:200
        c0 = 0.5 * (lo + hi)
        flag, _, _, _ = simulate(c0, k0)
        flag == :high ? (hi = c0) : (lo = c0)
        hi - lo < tol && break
    end
    return c0
end

k0 = 0.5 * kstar
c0 = shoot(k0)
println("(b) k0 = $(round(k0, digits=5)),  c0 = $(round(c0, digits=5))")

flag, kpath, cpath, tstop = simulate(c0, k0)
println("    simulation flag: $flag")

# The saddle path is unstable numerically, so after tstop the path drifts away
# from the true solution. Keep the valid part and fill the rest with the steady state.
kpath = kpath[1:tstop]; cpath = cpath[1:tstop]
if tstop < T + 1
    kpath = vcat(kpath, fill(kstar, T + 1 - tstop))
    cpath = vcat(cpath, fill(cstar, T + 1 - tstop))
end

# Δk = 0 curve: c = f(k) - δ k
kgrid = range(0.01, 1.1 * kstar, length = 300)
dk0   = f.(kgrid) .- δ .* kgrid

p1 = plot(kgrid, dk0, lw = 2, label = "Δk = 0 curve",
          xlabel = "Capital, k", ylabel = "Consumption, c",
          title = "Saddle path and Δk = 0 curve", legend = :topleft)
plot!(p1, kpath, cpath, lw = 2, label = "Saddle path", marker = :circle, ms = 2)
scatter!(p1, [kstar], [cstar], label = "Steady state", color = :red, ms = 6)
scatter!(p1, [k0], [c0], label = "Initial condition", color = :green, ms = 6)
savefig(p1, "figures/saddle_path.png")

# ---------------------------------------------------------------
# (c) Time series and half-life
# ---------------------------------------------------------------
tt = 0:T
p2 = plot(tt, kpath, lw = 2, label = "k_t",
          xlabel = "Time (years)", ylabel = "Capital, k",
          title = "Capital over time", legend = :bottomright)
hline!(p2, [kstar], ls = :dash, label = "k*")
savefig(p2, "figures/capital_series.png")

p3 = plot(tt, cpath, lw = 2, label = "c_t",
          xlabel = "Time (years)", ylabel = "Consumption, c",
          title = "Consumption over time", legend = :bottomright)
hline!(p3, [cstar], ls = :dash, label = "c*")
savefig(p3, "figures/consumption_series.png")

# Half-life: time until k_t reaches the midpoint between k0 and k*
khalf = k0 + 0.5 * (kstar - k0)
idx   = findfirst(x -> x >= khalf, kpath)       # position in vector (time = idx - 1)
frac  = (khalf - kpath[idx-1]) / (kpath[idx] - kpath[idx-1])
halflife = (idx - 2) + frac
println("(c) Half-life: first year with k_t >= midpoint = $(idx - 1);  interpolated = $(round(halflife, digits=2)) years")

# ---------------------------------------------------------------
# (d) Policy function k_{t+1} = g(k_t)
# ---------------------------------------------------------------
k0s = range(0.1 * kstar, 1.5 * kstar, length = 40)
k1s = Float64[]
for k0i in k0s
    ci = shoot(k0i)
    push!(k1s, f(k0i) + (1 - δ) * k0i - ci)
end

p4 = plot(k0s, k1s, lw = 2, label = "Policy function k_{t+1} = g(k_t)",
          xlabel = "Capital today, k_t", ylabel = "Capital tomorrow, k_{t+1}",
          title = "Policy function", legend = :topleft)
plot!(p4, k0s, k0s, ls = :dash, label = "45° line")
scatter!(p4, [kstar], [kstar], color = :red, ms = 6, label = "Steady state")
savefig(p4, "figures/policy_function.png")

println("Figures saved in ./figures")
