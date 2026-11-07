# Math Graph Plotter

Type a math expression and see it as a live, animated graph.
Each free variable becomes a slider, so you can watch how a constant changes the shape.

## Run

Open `index.html` in a browser. There is no build step. It needs internet access to load math.js and KaTeX from a CDN.

## Writing expressions

| You type | Meaning |
|---|---|
| `y = a*sin(b*x)` | Explicit curve. `a` and `b` become sliders. |
| `f(x) = a*x^2 + b*x + c` | Named function. Hovering shows `f(x)` and the slope `f′(x)`. |
| `T(n) = n*log2(n)` | Any argument name works as the horizontal variable. |
| `r = cos(k*theta)` | Polar curve. Set the θ range under *Display*. |
| `y = sin(x - t)` | `t` is time and animates automatically. |

Supported: `+ - * / ^`, implicit multiplication (`2x`), `sin cos tan exp log log2 sqrt abs sign …`, and the constants `pi` and `e`.

## Learning aids

- **▶ on a slider** sweeps that parameter back and forth.
- **Ghost trails** keep faded copies of earlier frames, so you can see how the curve moves.
- **Tangent on hover** draws the tangent line and shows the slope at the cursor.
- **Plot derivative** draws f′(x) as a dashed curve. The symbolic derivative also appears under the selected expression.
- **Copy share link** puts the whole state (expressions, sliders, view) in the URL.

## Physics simulations (`sim.html`)

Some equations (Maxwell, Navier–Stokes) are *partial differential equations*. They can't be plotted directly; they have to be solved step by step on a grid. `sim.html` has two live solvers. Each one shows its equations with color-coded terms that you can switch off or adjust.

**Fluid: Navier–Stokes** (Stable Fluids, Stam 1999, with MacCormack dye advection)
- *Stir a tank:* drag to push the fluid and drop dye.
- *Wind tunnel:* flow past a cylinder. Change ν to move between a smooth wake and a von Kármán vortex street. The Reynolds number is shown live.
- *Shear layer:* the Kelvin–Helmholtz instability.
- Toggle advection or pressure off to see what each term does. Views: dye, speed, vorticity, pressure, velocity arrows.

**Light: Maxwell** (FDTD / Yee scheme, 2D TMz, absorbing borders)
- Experiments: point source, two-source interference, double slit, refraction, lens, parabolic dish.
- Draw your own sources, plane waves, metal and glass. Adjust the wavelength and the glass refractive index.
- Views: electric field, time-averaged intensity, magnetic-field arrows. Slow-mo and single-step show the Faraday ↔ Ampère hand-off.

**Control: PID feedback** (960 Hz physics, controller runs every step)
- Three plants: *servo motor* (DC motor + arm with gravity), *drone altitude* (gravity, drag, wind), *balancing robot* (inverted pendulum on a cart, with an outer position loop).
- Sliders for Kp, Ki, Kd, plus presets that each show one lesson: P-only steady-state error, PD damping, PID, overshoot, integral windup, an unstable pendulum.
- A live block diagram (r → Σ → PID → plant → sensor), plots of the output and of each P/I/D contribution, and rise-time / overshoot / settling metrics.
- Real-world effects: actuator saturation, anti-windup, sensor noise (amplified by D), derivative kick, feed-forward, disturbances and a ⚡ Poke button.

## Files

- `index.html`: page layout
- `styles.css`: styling, light and dark mode
- `app.js`: parsing (math.js), sampling, canvas rendering, animation, interaction
- `sim.html`, `sim.css`, `sim.js`: simulation page shell (tabs, canvas, input routing)
- `sim-fluid.js`: Navier–Stokes solver and its panel
- `sim-maxwell.js`: Maxwell FDTD solver and its panel
- `sim-control.js`: PID controller, the three plants, plots and loop diagram
- `sim-ui.js`: shared controls and color maps
