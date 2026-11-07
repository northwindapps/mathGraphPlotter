# Math Graph Plotter & Simulations

Interactive, animated visualizations for the math behind computer science and engineering: functions, differential equations, feedback control and signal transforms.

The project started from a simple observation: formulas full of variables and constants are hard to grasp as static symbols on a page. Watching them **move** while you change one constant at a time makes them much easier to understand. Every tool here follows the same rule: show the equation, color-code its terms, let you change each one, and show the effect immediately.

Everything runs in the browser. There is no build step and nothing to install.

---

## Quick start

1. Open **`index.html`** for the graph plotter, or **`sim.html`** for the simulations. The two pages link to each other.
2. An internet connection is needed the first time, to load [math.js](https://mathjs.org) and [KaTeX](https://katex.org) from a CDN.

Optionally, serve the folder locally instead of opening the files directly:

```sh
python3 -m http.server 8000   # then visit http://localhost:8000
```

---

## 1. Graph plotter (`index.html`)

Type an expression and it is drawn immediately. **Every free letter becomes a slider**, and **`t` is time**, so it animates.

| You type | What happens |
|---|---|
| `y = A*sin(k*x - w*t)` | A traveling wave, with sliders for A, k and ω |
| `f(x) = a*x^2 + b*x + c` | A named function; hover to see f(x), the slope f′(x) and the tangent line |
| `T(n) = n*log2(n)` | Any argument name works as the horizontal variable |
| `r = cos(k*theta)` | A polar curve (θ range adjustable) |

**Learning aids**
- ▶ on any slider sweeps that parameter back and forth.
- *Ghost trails* leave faded copies of earlier frames.
- *Plot derivative* draws f′(x), and its symbolic formula appears under the expression.
- Every expression is also shown in textbook notation (KaTeX).
- *Copy share link* stores the whole state in the URL.

**Built-in examples:** traveling and standing waves, beats, quadratic, sigmoid, normal distribution, damped oscillation, Big-O growth rates, Amdahl's law, a Fourier square wave, polar rose, spiral.

**Supported syntax:** `+ - * / ^`, implicit multiplication (`2x`), `sin cos tan exp log log2 sqrt abs sign …`, and the constants `pi` and `e`.

---

## 2. Simulations (`sim.html`)

Some of the most important equations, such as Maxwell's and Navier–Stokes, are **partial differential equations**. They can't be plotted like `y = f(x)`; they have to be *solved* numerically, one small time step at a time. The simulation page has six tabs, and each one shows its governing equations with color-coded terms tied to the controls.

### Fluid: Navier–Stokes
∂u/∂t + (u·∇)u = −∇p/ρ + ν∇²u + f, with ∇·u = 0

- **Scenarios:**
  - *Stir a tank:* drag to push the fluid and drop dye.
  - *Wind tunnel:* flow past a cylinder; lower the viscosity and a **von Kármán vortex street** forms. The Reynolds number is shown live.
  - *Shear layer:* two layers sliding past each other roll up into **Kelvin–Helmholtz** waves.
- **Term toggles:** switch off advection (the nonlinear term) or pressure/incompressibility and watch the fluid stop behaving like water.
- **Views:** dye, speed, vorticity, pressure, velocity arrows.

### Light: Maxwell's equations
- **Experiments:** point source, two-source interference, **double slit**, refraction, a **lens** focusing light, a **parabolic dish** forming a beam.
- **Drawing tools:** paint metal and glass, and place point sources or plane-wave sources. Adjust the wavelength and the glass refractive index.
- **Views:** electric field E, time-averaged intensity, magnetic-field arrows B.
- **Slow-mo and single-step** show the Faraday ↔ Ampère–Maxwell hand-off that *is* the light wave.

### Control: PID feedback
u(t) = Kp·e + Ki·∫e dt + Kd·de/dt

- **Three systems to control:**
  - **Servo motor:** a DC motor arm with gravity, which shows why the I term is needed.
  - **Drone altitude:** gravity, drag and wind; shows feed-forward and integral windup.
  - **Balancing robot:** an inverted pendulum on a cart, unstable by nature, which shows why the D term is needed. It also shows the counterintuitive "back up first" move.
- **Live displays:**
  - a block diagram (r → Σ → PID → plant → sensor) with values on each wire;
  - plots of the output against its target, and of each P, I and D contribution;
  - rise time, overshoot, settling time and steady-state error.
- **Presets**, each showing one lesson: P only, PD, PID, too aggressive, windup trap, too weak.
- **Real-world effects:** actuator saturation, anti-windup, sensor noise (amplified by D), derivative kick, disturbances, and a ⚡ Poke button.

### Fourier: series & transform
- **Build from sines:** rotating circles (epicycles) add up to a square, sawtooth, triangle, pulse or **hand-painted** wave, with the spectrum below. Shows the Gibbs overshoot.
- **Winding machine:** the signal is wrapped around a circle at frequency f, and the center of mass of the wrapped curve *is* X(f). It jumps away from zero exactly when f matches a frequency in the signal.
- **Draw with circles:** draw any closed shape and spinning circles redraw it, which is the same idea behind JPEG and MP3 compression.

### Laplace: the s-plane
- **Each point is a signal:** every point s = σ + iω corresponds to a growing or decaying oscillation e^{st}. Hover to see it.
- **Drag poles (×) and zeros (○):** H(s), the impulse response, the step response and the frequency response all update live. Presets: RC filter, spring, heavy damping, oscillator, unstable, notch.
- **3D landscape of |H(s)|:** poles push the surface up like tent poles, and the slice along the jω axis is the frequency response. **The Fourier transform is one slice of the Laplace transform.**

### Circuits: RLC & differential equations
L·q″ + R·q′ + q/C = V(t), which is Kirchhoff's voltage law around the loop.

- **Five circuits:**
  - **RC:** charging a capacitor; τ = RC.
  - **RL:** current building up in a coil; τ = L/R.
  - **LC:** endless oscillation; ω₀ = 1/√(LC).
  - **Series RLC:** the general second-order system.
  - **Parallel RLC:** the "tank circuit", driven by a current source. It follows KCL, C·v′ + v/R + (1/L)∫v dt = I(t), and the panel shows branch currents splitting and rejoining at the junctions.
- **Series vs. parallel duality:** swapping voltage ↔ current, L ↔ C, R ↔ 1/R and KVL ↔ KCL turns one circuit into the other. In parallel a *larger* R gives *less* damping (critical at R = ½√(L/C)), and resonance peaks in voltage instead of current.
- **Animated schematic:**
  - current flows as moving charges;
  - the resistor glows with heat;
  - + and − charges pile up on the capacitor plates;
  - the inductor's magnetic field pulses.
- **Each component as a term:** the resistor's drop depends on q′, the inductor's on q″, the capacitor's on q. Each has its own slider and explanation.
- **Live solution:** the characteristic equation Ls² + Rs + 1/C = 0, its roots on a mini s-plane, and the label *underdamped / critically damped / overdamped / undamped*, along with the R value that gives critical damping.
- **Sources:** DC with a switch, square wave, AC sine. With the sine source you appear as a point on the **resonance curve**.
- **Plots:** voltages (which always add up to the source), current, and the energy sloshing between the capacitor and the inductor.
- **Mechanical twin:** a mass, spring and damper beside the circuit, driven by the same equation (L ↔ m, R ↔ c, 1/C ↔ k, q ↔ x).

---

## How it works

| Topic | Method |
|---|---|
| Expression parsing & derivatives | math.js parse + symbolic `derivative`, with a numerical fallback |
| Curve drawing | Adaptive sampling; asymptotes are detected by bisection, so `tan x` and `1/x` break cleanly |
| Fluid | *Stable Fluids* (J. Stam, SIGGRAPH 1999): semi-Lagrangian advection, implicit diffusion, Gauss–Seidel pressure projection. Dye uses MacCormack advection with a limiter to stay sharp |
| Light | FDTD on a Yee grid (K. Yee, 1966), 2D TMz polarisation, Courant number 0.5, graded absorbing border, PEC metal, dielectric glass |
| Control | 960 Hz physics with semi-implicit Euler; PID with a filtered derivative, conditional-integration anti-windup and optional derivative-on-measurement; full nonlinear cart-pole dynamics |
| Fourier | Direct DFT of 512 samples per period; complex DFT of a 256-point arc-length-resampled path for the drawings |
| Laplace | Exact impulse and step responses from partial-fraction residues, h(t) = Σ rₖ e^{pₖt}; gain normalized to unit DC gain |
| Circuits | 4th-order Runge–Kutta on (q, i), 20 sub-steps per frame; resonance curve from the impedance, \|I\| = V/\|Z\| |

**Code layout**

```
index.html, styles.css, app.js     graph plotter
sim.html, sim.css, sim.js          simulation shell (tabs, canvas, input routing, loop)
sim-ui.js                          shared controls and color maps
sim-fluid.js                       Navier–Stokes
sim-maxwell.js                     Maxwell FDTD
sim-control.js                     PID and the three plants
sim-fourier.js                     Fourier series, winding machine, drawing with circles
sim-laplace.js                     s-plane, responses, 3D landscape
sim-circuit.js                     RC / RL / LC / series & parallel RLC circuits and the mechanical twin
```

Each simulation is a self-contained module with the same small interface: `mount`, `resize`, `frame`, `pointer`. Adding a new tab means adding one file and one button.

---

## Verification

The numerics were checked against known answers, not only by eye:

- **Fourier:** square-wave coefficients match 4/π, 0, 4/(3π), … to four decimals; the winding machine finds the correct peaks and heights.
- **Laplace:** impulse responses match closed-form solutions (first-order, underdamped, overdamped, unstable) to machine precision; a zero on the jω axis gives exactly zero gain there.
- **Circuits:** the RC, RL, LC, underdamped series RLC and parallel RLC responses match their closed-form solutions to about 10⁻¹² relative error; LC energy is conserved to 10⁻¹³ over 10 s. Driving at f₀ gives exactly V/R (series current) and I·R (parallel voltage).
- **Control:** every preset behaves as its label says. P-only error matches theory (drone: 9.81 N / Kp ≈ 1.2 m); turning anti-windup off raises overshoot from about 30% to about 90%; "Too weak" really falls over.
- **Fluid and light:** long runs show no NaNs or blow-ups; vortex shedding, Kelvin–Helmholtz roll-up, double-slit fringes, lens focusing and dish collimation all appear as expected.
- **Performance:** the fluid takes about 13 ms per frame, everything else under 2 ms, so all tabs should run at 60 fps.

Rendering was checked with headless-Chrome screenshots. Mouse and touch interaction (dragging, painting, drawing) still needs hands-on testing in a real browser.

## Known limitations

- The simulations are 2D. The fluid grid is coarse (about 150×125 cells), so the method itself adds some smoothing; real turbulence would need finer grids and GPU acceleration.
- The plotter doesn't yet support `x = …`, parametric or implicit curves.

## Ideas for next steps

Convolution and filtering, the z-transform, Bode and root-locus plots, a GPU (WebGL) port for higher-resolution simulations, heat and Schrödinger equations.

---

## Author

**Designed and written by Claude** (Claude Opus 5.5, an AI model by [Anthropic](https://www.anthropic.com)), working in Claude Code at the request of, and with direction from, the project owner. The owner set the goals and topics: making the math of CS and engineering visual, then adding Maxwell, Navier–Stokes, PID control, Fourier, Laplace and RLC circuits.

Claude chose the technology (plain JavaScript and Canvas, no build step), designed the interfaces and explanations, implemented every solver and visualization, and verified the numerics described above.

**Third-party libraries:** [math.js](https://mathjs.org) (Apache-2.0) and [KaTeX](https://katex.org) (MIT), both loaded from CDNs.

**Methods credited to their originators:** Jos Stam (*Stable Fluids*, 1999), Kane S. Yee (FDTD, 1966), and the classical work of Fourier, Laplace, Maxwell, Navier and Stokes.
