import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt



# Compatibility shim: works on NumPy 1.x (Databricks) and 2.x
trapz = np.trapz if hasattr(np, 'trapz') else np.trapezoid

# ── PID controller parameters ──────────────────────────────────
#
#   Kr   : proportional gain
#   tauI : integral time (hr)  — smaller = faster offset removal
#   tauD : derivative time (hr) — set to 0 for PI only

Kr   = -0.5   # proportional gain  (m3/hr per degC error)
tauI = 10.0    # integral time      (hr)
tauD =  0.0    # derivative time    (hr)

# ── Coolant flow limits ────────────────────────────────────────
F_min = 0.000   # pump off (cannot heat)
#F_max = 0.100   # maximum pump capacity  (m3/hr) (100 L/hr)
F_max = 0.100 #changed to 500 L/hr

# ── Disturbance ───────────────────────────────────────────────
disturbance_dT = 3.0   # degC step added to wort at 40% of run

print(f'PID: Kr={Kr}, tauI={tauI}, tauD={tauD}')
print(f'Flow limits: [{F_min}, {F_max}] m3/hr')

# ── Simulation timing ─────────────────────────────────────────
duration      = 336
t_disturbance = 0.40 * duration   # 134.4 h

# ── Setpoint profile (same as disturbance notebook) ──────────
t_base   = np.linspace(0, duration, 1500)
Tset_arr = np.zeros(len(t_base))
for i, ti in enumerate(t_base):
    if   ti < 48:   Tset_arr[i] = 18
    elif ti < 168:  Tset_arr[i] = 20
    elif ti < 240:  Tset_arr[i] = 22
    elif ti < 300:  Tset_arr[i] = 18
    else:           Tset_arr[i] = 18 - (16/36) * (ti - 300)

Tset_func  = interp1d(t_base, Tset_arr,                       fill_value='extrapolate')
dTset_func = interp1d(t_base, np.gradient(Tset_arr, t_base),  fill_value='extrapolate')

# ── Physical properties (identical to disturbance notebook) ───
OG               = 1.048
yeast_pitch_rate = 0.75
yeast_density    = 40e-12       # g/cell
volume           = 1200         # L  (wort)
vessel_volume    = 1500         # L
vessel_volume_m3 = vessel_volume / 1000
temperature      = 18           # degC  (initial)
temperature_K    = temperature + 273.15

k_s  = 15.3;  k_e  = 6.31;   S_min = 13.1
g_x  = 0.0167; k_v  = 0.651;  k_co2 = 3.71
a_MB = 0.4;   b_MB = -1.1;   c_MB  = 0.025;  d_MB = -0.056

degP         = (-616.868) + (1111.14*OG) - (630.272*OG**2) + (135.997*OG**3)
wort_density_MB = OG * 1000
s_calc       = (degP/100) * wort_density_MB
yeast_req    = yeast_pitch_rate * (volume * 1000) * degP * 1e6
mass_yeast   = yeast_req * yeast_density
yeast_conc   = mass_yeast / volume

T_water_in   = 10               # degC  (coolant inlet temperature)
T_water_in_K = T_water_in + 273.15
water_density = (999.80 + 992.30) / 2
water_Cp      = 4.1789          # kJ/kg/K
wort_density  = (1048 + 1005) / 2
wort_Cp       = 4.0585          # kJ/kg/K

U      = 50                     # overall heat transfer coeff  (W/m2/K)
U_hr   = U * 3600               # J/m2/K/hr
delta_H = -587                  # kJ/kg  (heat of fermentation)
Area   = 0.3 * 14.137 / 2      # m2  (jacket heat transfer area)
V_wort_m3 = volume / 1000
Cooling_Jacket_Volume = vessel_volume_m3 * 0.05

print('Setup complete — all values identical to disturbance notebook.')
print(f'  Coolant inlet: {T_water_in} degC  |  U={U} W/m2/K  |  A={Area:.3f} m2')
print(f'  Jacket volume: {Cooling_Jacket_Volume:.4f} m3')

def ODE_with_PID(t, state):

    x, s, e, co2, vdk, T, Tc, I = state

    # ── PID controller ────────────────────────────────────────
    # Error between setpoint and wort temperature
    error = Tset_func(t) - (T - 273.15)

    # PID output = coolant flow rate (m3/hr)
    # D term uses dTset/dt to avoid a spike when the setpoint steps
    F_PID = Kr * error + (Kr / tauI) * I + Kr * tauD * dTset_func(t)

    # Clamp to pump limits
    F_coolant_m3h = float(np.clip(F_PID, F_min, F_max))

    # Anti-windup: freeze integral when pump is saturated
    saturated = (F_PID <= F_min) or (F_PID >= F_max)
    dI_dt = 0.0 if saturated else error

    # ── Mass balances (unchanged from disturbance notebook) ───
    mu_max_T = a_MB * np.log(max(T - 273.15, 0.1)) + b_MB
    r_vdk_T  = c_MB * np.log(max(T - 273.15, 0.1)) + d_MB
    mu       = mu_max_T * (1 - S_min / s) if s >= S_min else 0.0

    dx_dt   = (mu * x) - (g_x * x)
    ds_dt   = -k_s * mu * x
    de_dt   =  k_e * mu * x
    dco2_dt =  k_co2 * mu * x
    dvdk_dt =  k_v * mu * x - r_vdk_T * vdk

    # ── Wort heat balance (unchanged from disturbance notebook) ─
    wort_temp_term1 = (ds_dt * delta_H
                       / (wort_density * wort_Cp))

    wort_temp_term2 = (Area * U_hr
                       / (V_wort_m3 * wort_density * wort_Cp * 1000)
                       * (T - Tc))

    dT_dt = wort_temp_term1 - wort_temp_term2

    # ── Jacket heat balance (unchanged from disturbance notebook) 
    # F_coolant_m3h is now the PID output instead of a fixed value
    water_temp_term1 = (F_coolant_m3h / Cooling_Jacket_Volume) * (T_water_in_K - Tc)
    water_temp_term2 = (Area * U_hr
                        / (Cooling_Jacket_Volume * water_density * water_Cp * 1000)
                        * (T - Tc))
    dTc_dt = water_temp_term1 + water_temp_term2

    return [dx_dt, ds_dt, de_dt, dco2_dt, dvdk_dt, dT_dt, dTc_dt, dI_dt]

print('ODE defined.')

# ── Initial conditions ────────────────────────────────────────
# Same as disturbance notebook, plus I = 0 for the integral state
y0 = [yeast_conc, s_calc, 0.0, 0.0, 0.0,
      temperature_K,    # wort at 18 degC
      T_water_in_K,     # jacket at coolant inlet temp
      0.0]              # integral state = 0

# ── Time arrays ───────────────────────────────────────────────
n1     = int(2000 * t_disturbance / duration)
n2     = 2000 - n1
t_seg1 = np.linspace(0,             t_disturbance, n1)
t_seg2 = np.linspace(t_disturbance, duration,      n2)

# ── Segment 1: normal operation with PID ─────────────────────
sol1 = solve_ivp(ODE_with_PID, (0, t_disturbance),
                 y0, t_eval=t_seg1, method='BDF', max_step=0.1)
if not sol1.success:
    raise RuntimeError(f'Segment 1 failed: {sol1.message}')

# ── Apply disturbance: +3 degC to wort (state index 5) ───────
y_after = sol1.y[:, -1].copy()
T_before = y_after[5] - 273.15
y_after[5] += disturbance_dT

print(f'Wort temp just before disturbance : {T_before:.2f} degC')
print(f'Wort temp after +{disturbance_dT} degC step    : {y_after[5]-273.15:.2f} degC')

# ── Segment 2: PID rejects the disturbance ────────────────────
sol2 = solve_ivp(ODE_with_PID, (t_disturbance, duration),
                 y_after, t_eval=t_seg2, method='BDF', max_step=0.1)
if not sol2.success:
    raise RuntimeError(f'Segment 2 failed: {sol2.message}')

print('\nBoth segments solved successfully.')

# ── Stitch results ────────────────────────────────────────────
t_all = np.concatenate([sol1.t,    sol2.t[1:]])
y_all = np.concatenate([sol1.y,    sol2.y[:, 1:]], axis=1)

T_wort  = y_all[5] - 273.15   # wort temperature (degC)
T_jack  = y_all[6] - 273.15   # jacket temperature (degC)
I_state = y_all[7]             # integral state
Tset    = Tset_func(t_all)     # setpoint (degC)
error   = Tset - T_wort        # tracking error

# Reconstruct coolant flow history from PID equation
F_cool = np.clip(
    Kr * error + (Kr / tauI) * I_state + Kr * tauD * dTset_func(t_all),
    F_min, F_max
)

# ── Performance metrics ───────────────────────────────────────
i_d   = np.searchsorted(t_all, t_disturbance)
i_d1  = np.searchsorted(t_all, t_disturbance + 1)
i_d5  = np.searchsorted(t_all, t_disturbance + 5)
i_d10 = np.searchsorted(t_all, t_disturbance + 10)

peak_T = T_wort[i_d:].max()
t_peak = t_all[i_d + np.argmax(T_wort[i_d:])]
IAE    = trapz(np.abs(error[i_d:]), t_all[i_d:]) #Integral Absolute Error

print(f'\n--- Post-disturbance performance ---')
print(f'  Peak wort temperature   : {peak_T:.2f} degC  at t = {t_peak:.1f} h')
print(f'  Max deviation from Tset : {(-error[i_d:]).max():.2f} degC')
print(f'  Temp 1 h after dist.    : {T_wort[i_d1]:.2f} degC  (setpoint {Tset[i_d1]:.1f} degC)')
print(f'  Temp 5 h after dist.    : {T_wort[i_d5]:.2f} degC')
print(f'  Temp 10 h after dist.   : {T_wort[i_d10]:.2f} degC')
print(f'  IAE (post-disturbance)  : {IAE:.1f} degC.h')

fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
fig.suptitle(
    f'Beer Fermentation — PID Control\n'
    f' Kr = {Kr}, tau_I = {tauI}, tau_D = {tauD}'
    f'+{disturbance_dT} degC disturbance at t = {t_disturbance:.0f} h',
    fontsize=12, fontweight='bold'
)

# ── Panel 1: Wort and jacket temperature ─────────────────────
ax = axes[0]
ax.plot(t_all, Tset,   'k--', lw=1.5, label='Tset')
ax.plot(t_all, T_wort, color='#E8593C', lw=2.0, label='Wort temp (controlled)')
ax.plot(t_all, T_jack, color='steelblue', lw=1.3, ls='--', alpha=0.8, label='Jacket temp')
ax.axvline(t_disturbance, color='grey', ls=':', lw=1.3,
           label=f'Disturbance at t = {t_disturbance:.0f} h')
ax.annotate(f'+{disturbance_dT:.0f} degC step',
            xy=(t_disturbance, T_before + disturbance_dT),
            xytext=(t_disturbance + 18, T_before + disturbance_dT + 0.8),
            arrowprops=dict(arrowstyle='->', color='#333'), fontsize=9)
ax.set_ylabel('Temperature (degC)', fontsize=11)
ax.set_title('Wort Temperature vs Setpoint', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

# ── Panel 2: Temperature error ────────────────────────────────
window = 15
plotline = t_disturbance + window

ax = axes[1]
ax.fill_between(t_all, error, 0, where=(error > 0),
                color='steelblue', alpha=0.3, label='Wort too cold (below Tset)')
ax.fill_between(t_all, error, 0, where=(error < 0),
                color='#E8593C',  alpha=0.3, label='Wort too hot  (above Tset)')
ax.plot(t_all, error, color='#333', lw=1.2)
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.axvline(plotline, color = 'grey')
ax.axvline(t_disturbance, color='grey', ls=':', lw=1.3)
ax.set_ylabel('Error = Tset - Twort (degC)', fontsize=11)
ax.set_title('Temperature Error', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

# ── Panel 3: Coolant flow rate (manipulated variable) ─────────
ax = axes[2]
ax.plot(t_all, F_cool * 1000, color='steelblue', lw=2.0,
        label='Coolant flow (PID output)')
ax.axhline(F_max * 1000, color='red',  lw=0.8, ls=':', label=f'F_max = {F_max*1000:.0f} L/hr')
ax.axhline(F_min * 1000, color='navy', lw=0.8, ls=':', label=f'F_min = {F_min*1000:.0f} L/hr')
ax.axvline(t_disturbance, color='grey', ls=':', lw=1.3)
ax.set_ylim(-1, F_max * 1000 * 1.2)
ax.set_ylabel('Coolant Flow Rate (L/hr)', fontsize=11)
ax.set_xlabel('Time (h)', fontsize=11)
ax.set_title('Manipulated Variable — Coolant Flow Rate', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
#plt.savefig('/mnt/user-data/outputs/PID_Wort_Control.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Change these two lines ────────────────────────────────────
sweep_param  = 'Kr'
sweep_values = [-0.5, -2.0, -5.0, -10.0]
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f'Effect of {sweep_param} on disturbance rejection',
             fontsize=12, fontweight='bold')
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sweep_values)))

for val, col in zip(sweep_values, colors):

    _Kr   = val  if sweep_param == 'Kr'   else Kr
    _tauI = val  if sweep_param == 'tauI' else tauI
    _tauD = val  if sweep_param == 'tauD' else tauD

    def ode_sweep(t, state, k=_Kr, ti=_tauI, td=_tauD):
        x, s, e, co2, vdk, T, Tc, I = state
        err  = Tset_func(t) - (T - 273.15)
        F_p  = k * err + (k / ti) * I + k * td * dTset_func(t)
        F_c  = float(np.clip(F_p, F_min, F_max))
        dI   = 0.0 if (F_p <= F_min or F_p >= F_max) else err
        mu_max = a_MB * np.log(max(T-273.15, 0.1)) + b_MB
        r_vdk  = c_MB * np.log(max(T-273.15, 0.1)) + d_MB
        mu   = mu_max * (1 - S_min/s) if s >= S_min else 0.0
        ds   = -k_s * mu * x
        dT   = (ds*delta_H/(wort_density*wort_Cp)) \
               - (Area*U_hr/(V_wort_m3*wort_density*wort_Cp*1000)) * (T - Tc)
        dTc  = (F_c/Cooling_Jacket_Volume)*(T_water_in_K - Tc) \
               + (Area*U_hr/(Cooling_Jacket_Volume*water_density*water_Cp*1000)) * (T - Tc)
        return [mu*x-g_x*x, ds, k_e*mu*x, k_co2*mu*x, k_v*mu*x-r_vdk*vdk, dT, dTc, dI]

    s1 = solve_ivp(ode_sweep, (0, t_disturbance), y0,
                   t_eval=t_seg1, method='BDF', max_step=0.1)
    yd = s1.y[:, -1].copy();  yd[5] += disturbance_dT
    s2 = solve_ivp(ode_sweep, (t_disturbance, duration), yd,
                   t_eval=t_seg2, method='BDF', max_step=0.1)

    t_sw  = np.concatenate([s1.t,    s2.t[1:]])
    T_sw  = np.concatenate([s1.y[5], s2.y[5,1:]]) - 273.15
    I_sw  = np.concatenate([s1.y[7], s2.y[7,1:]])
    err_sw = Tset_func(t_sw) - T_sw
    F_sw  = np.clip(_Kr*err_sw + (_Kr/_tauI)*I_sw
                    + _Kr*_tauD*dTset_func(t_sw), F_min, F_max)

    axes[0].plot(t_sw, T_sw,      color=col, lw=1.8, label=f'{sweep_param} = {val}')
    axes[1].plot(t_sw, F_sw*1000, color=col, lw=1.8)

axes[0].plot(t_all, Tset, 'k--', lw=1.5, label='Tset')
axes[0].axvline(t_disturbance, color='grey', ls=':', lw=1.2)
axes[0].set_ylabel('Wort Temperature (degC)', fontsize=11)
axes[0].legend(fontsize=9, loc='upper right')
axes[0].grid(alpha=0.3)

axes[1].axhline(F_max*1000, color='red',  lw=0.8, ls=':', label='F_max')
axes[1].axhline(F_min*1000, color='navy', lw=0.8, ls=':', label='F_min')
axes[1].axvline(t_disturbance, color='grey', ls=':', lw=1.2)
axes[1].set_ylabel('Coolant Flow Rate (L/hr)', fontsize=11)
axes[1].set_xlabel('Time (h)', fontsize=11)
axes[1].legend(fontsize=9, loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
#plt.savefig('/mnt/user-data/outputs/PID_Wort_Sweep.png', dpi=150, bbox_inches='tight')
plt.show()

#-------plot of mass balances:-----------

ethanol = y_all[2]   # ethanol concentration
vdk     = y_all[4]   # VDK concentration
ethanol_abv = (ethanol / 789) * 100 
vdk_ppb = vdk * 1000

plt.plot(t_all, ethanol_abv)
plt.xlabel("Time (hours) ")
plt.ylabel("Ethanol Concentration (ABV %)")
plt.title("Ethanol Concentration with Time")
plt.show()

plt.plot(t_all, vdk_ppb)
plt.xlabel("Time (hours)")
plt.ylabel("VDK concentration (ppb)")
plt.title("VDK concentration with Time")
plt.axhline(60, color='red',   linestyle='--', label='60 ppb')
plt.legend()
plt.show()

#----------plot of jacket and wort temperature:----------------
T_wort = y_all[5] - 273.15
T_jacket = y_all[6] - 273.15

plt.figure(figsize=(10, 5))

plt.plot(t_all, T_wort, label='Wort Temperature',   color='#E8593C', lw=2)
plt.plot(t_all, T_jack, label='Jacket Temperature', color='steelblue', lw=2, linestyle='--')

plt.axvline(t_disturbance, color='grey', linestyle=':', label='Disturbance')

plt.xlabel('Time (h)')
plt.ylabel('Temperature (°C)')
plt.title('Wort and Jacket Temperature vs Time')
plt.legend()
plt.grid(alpha=0.3)

plt.show()

#--------edit:------
threshold = 50

# Find index of peak VDK
i_peak = np.argmax(vdk_ppb)

# Look only after the peak
below_idx = np.where(vdk_ppb[i_peak:] < threshold)[0]

if len(below_idx) > 0:
    t_below_50 = t_all[i_peak + below_idx[0]]
    print(f"VDK drops below 50 ppb after peak at t = {t_below_50:.2f} hours")
else:
    print("VDK never drops below 50 ppb after the peak")

# what is the maximum overshoot after the disturbance

i_window = np.searchsorted(t_all, t_disturbance + window)
# Compute overshoot = error at that exact time
overshoot = Tset[i_window] - T_wort[i_window]

print(f"Overshoot at t = {window} h after disturbance = {overshoot:.2f} °C")

#-----tuning a PID controller:-----------
# find values of K_r, tau_I and tau_D which give you:
# fast response, minimal overshoot, stable behaviour, good disturbance rejection
# start simple: PI controller, set tauD = 0.0, derivate control is usually noisy, unnecessarily slow for thermal systems like fermentation

# transfer function from literature (Fergus)



 