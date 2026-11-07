# Math Graph Plotter & Simulations

**▶ Try it live: [northwindapps.github.io/mathGraphPlotter](https://northwindapps.github.io/mathGraphPlotter/)** (graph plotter) · **[all 12 simulation topics](https://northwindapps.github.io/mathGraphPlotter/sim.html)**. It runs in any browser, on a computer, tablet or phone, with nothing to install.

Interactive, animated visualizations for the math behind computer science and engineering: functions, differential equations, feedback control, signal transforms, linear algebra, probability and machine learning.

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

Some of the most important equations, such as Maxwell's and Navier–Stokes, are **partial differential equations**. They can't be plotted like `y = f(x)`; they have to be *solved* numerically, one small time step at a time. The simulation page has twelve tabs, and each one shows its governing equations with color-coded terms tied to the controls.

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
- **3D plane wave view:** the textbook picture of light, live and rotatable.
  - E (red) and B (blue) are perpendicular to each other and to the direction of travel, and in phase.
  - Optional Poynting arrows show the energy flowing forward.
  - **Polarization:** vertical, tilted, or circular (E traces a corkscrew, the Fourier helix again).
  - A **polarizer** filter passes only the component along its axis; the readout shows Malus's law, I = I₀cos²θ.
  - Camera buttons fly to "look along the beam", side and top views.

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
- **3D: time ↔ frequency:** the component sines are laid out along a frequency axis. From the **front** they stack into the signal (the time domain); rotate to the **side** and each wave collapses to a bar, which is the spectrum (the frequency domain). Buttons animate the camera between the two views, and a moving slice shows the sum at one instant.
- **3D: complex helix:** e^{iωt} drawn as a corkscrew in (time, Re, Im). Its shadows are cos (floor), sin (back wall) and the unit circle (looking down the time axis).
  - Show e^{−iωt} too: the two opposite spins average to a flat, real cosine, which is why spectra have ±f.
  - Add decay σ: the helix becomes the growing or shrinking spiral e^{st} that the Laplace transform is built from.

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

### Learning: how ML weights move
Models have millions of weights, but the dynamics of training can be seen in two dimensions, and the field itself studies them through 2D slices.

- **Fit a line:** y = w·x + b has exactly two weights. The data view (the line, with red error bars) and the loss surface L(w, b) update together as gradient descent walks downhill; a green arrow shows −∇L and a ★ marks the exact least-squares answer. Drag, add or delete points and the whole surface reshapes. Divergence starts exactly at the predicted learning rate η = 2/λmax.
- **Optimizer race:** plain gradient descent, momentum (the "heavy ball", m·θ″ + c·θ′ = −∇L, the same equation as the Circuits tab) and Adam race on four surfaces:
  - a narrow valley, where gradient descent zig-zags;
  - the Rosenbrock "banana" valley;
  - a saddle point, where gradient descent stalls and Adam escapes;
  - a surface with many minima, where the starting point decides which valley each optimizer ends up in.
  - **3D terrain view:** the loss surface as a shaded, rotatable landscape with the optimizers rolling on it as balls with trails. The saddle really looks like a saddle, and momentum visibly rolls over bumps. The Rosenbrock surface uses log-scaled height so its steep walls fit.
- **Tiny neural network:** a 2 → H → 1 network learns circle, XOR, two-moons and spiral datasets.
  - Each hidden neuron is drawn as a small map showing the single straight edge it draws; the output bends those edges into a curved boundary.
  - Weights are drawn as edges whose thickness and color show their value.
  - A **2D slice of the full loss landscape** (along two random directions, as in Li et al. 2018) has the projected training path drawn on top.

- **Lagrange multipliers:** optimize f(x, y) while staying on a path g(x, y) = c. Drag the point along the path, or let it slide, and watch ∇f (gold), λ∇g (green) and the part of ∇f along the path (red, zero at an optimum).
  - **Hill & path:** the best point is where the path just touches a contour, not at a hilltop. Every tangency is marked, and sliding can stop at a merely *local* optimum.
  - **Fence:** the classic maximum-area rectangle; the answer is a square with λ = P/8.
  - **Ridge (L2) vs. lasso (L1):** the same loss with a circle vs. a diamond budget. Lasso's optimum lands on a corner, with a weight of exactly 0. This is why L1 regularization selects features.
  - **λ as a "shadow price":** the readout compares λ* with the measured change in the best value as the constraint loosens (df*/dc).
  - **3D view:** the path lifted onto the landscape.

### Images: denoising
A 256 × 256 grayscale image (a test card, printed text, or **your own photo**, which never leaves the browser), with three kinds of noise and a three-stage pipeline:

- **Noise:** Gaussian grain, salt & pepper (dead pixels), and periodic stripes (interference) with adjustable strength, frequency and angle.
- **1 · Median filter:** a vote among neighbours, so outliers never win. The classic cure for salt & pepper.
- **2 · Fourier mask (2D FFT):**
  - Stripes show up in the spectrum as a pair of bright dots. Click a dot to notch it out, or let **auto-find** do it.
  - Auto-find tells noise from real image detail by its sharpness: stripes covering the whole image make a needle-sharp peak, while real patterns (text lines, a checkerboard) make broader bumps.
  - There is also a low-pass radius, plus an ideal (sharp) option that shows Gibbs ringing.
- **3 · Smoothing:** box and Gaussian blur (convolution), bilateral (edge-preserving), and total variation, a convex optimization, min ½‖u − f‖² + λ‖∇u‖₁, solved live with Chambolle's algorithm.
- **Compare:** original vs. noisy vs. denoised, an optional error map, the spectrum of the noisy or the result image, and a PSNR score before → after.

### Linear algebra: matrices, eigenvectors, PCA
- **Matrix = transformation:** drag where the basis arrows î and ĵ land (the matrix's columns), or use presets: rotate, stretch, shear, reflect, squash, symmetric.
  - The whole grid, the unit square and an "F" transform with it; "Animate from the identity" shows the motion.
  - The determinant is the square's area: det = 0 squashes the plane flat, and a negative det flips it (the F becomes mirror-imaged).
- **Eigenvectors:**
  - A ring of vectors shows where each one goes; eigen-directions are drawn dashed with their λ.
  - Drag v until Av lines up with it ("Av = λv").
  - Rotations give complex λ and no real eigen-direction.
  - **Power iteration** (apply A again and again) converges to the dominant eigenvector, which is the idea behind PageRank.
- **PCA:** a data cloud with its covariance ellipse and principal axes.
  - Rotate a projection line and watch the projected variance; its maximum is exactly λ₁ along PC1 (a Lagrange problem: max wᵀΣw subject to ‖w‖ = 1 gives Σw = λw).
  - "Variance explained" is shown live, and "Collapse 2D → 1D" animates the dimensionality reduction.

### Probability: Bayes, CLT, Markov chains
- **Bayes' rule:** the classic medical-test puzzle drawn as 2,000 people.
  - Sliders control how common the disease is (the prior), the test's sensitivity and its false-positive rate.
  - A bar shows who is really sick among everyone who tested positive (16.1% at the classic settings, not 95%), alongside a natural-frequency tree and the formula with live numbers.
  - "Positive again" uses the last posterior as the new prior: belief updating.
- **Central limit theorem:** average n draws from a uniform, exponential, die, two-humped or unfair-coin distribution, thousands of times.
  - The histogram of averages becomes a bell curve N(μ, σ²/n), whatever the starting shape.
  - The readout compares the measured spread with σ/√n and shows the skewness shrinking toward 0.
- **Markov chains:** a walker hops between states (weather, a 4-page web, two weakly linked "islands").
  - Its visit fractions converge to the stationary distribution π, the eigenvector of P with λ = 1. The web-surfer chain with damping d *is* PageRank.
  - A second chart shows p_k = p_0·P^k flattening onto π, and |λ₂| sets how fast; the islands chain mixes slowly (|λ₂| = 0.98).

### Convolution: filters & kernels
- **Slide & sum:** flip the kernel g, slide it across the signal f, multiply, and add. The green overlap area is the output at that point, and the output curve draws itself; drag to move by hand.
  - **Signals:** rectangle, step, triangle, noisy wave, spikes.
  - **Kernels:** box average, Gaussian, the **RC circuit's impulse response** (step * RC = the capacitor-charging curve), an edge detector (a derivative), and an echo.
  - Cards connect it to linear systems (output = input * impulse response) and to probability (sums of random variables convolve; repeated convolution is the CLT).
- **Convolution theorem:** the same result computed directly (N² multiply-adds) and by multiplying FFT spectra, shown side by side and agreeing to 10⁻¹⁵. A blur is a low-pass because its spectrum is small at high frequencies.
- **Image kernels (CNN):** an editable 3×3 kernel with presets (identity, box, Gaussian, sharpen, Laplacian edges, Sobel ↔/↕, emboss) applied to the test card.
  - Hover a pixel to see its 9 multiply-adds worked out.
  - Cards explain that a CNN *learns* these 9 numbers, how to read a kernel (sum 1 keeps brightness, sum 0 detects change), and convolution vs. cross-correlation.

### Tensors: n-dimensional arrays and loops
- **Shapes & memory:** rank 0 to 4 tensors (scalar, vector, matrix, stack of matrices, batch), each shown above the **one flat array** that actually holds the numbers.
  - Hover any cell to see offset = Σ index·stride worked out.
  - **Transpose** swaps strides without moving data, so the view becomes non-contiguous. **Reshape** is free on contiguous data and needs a copy otherwise (`.view` vs. `.contiguous()`).
- **Operations = loops:** add, broadcasting, sum over an axis (reduction), matrix multiply (contraction) and outer product. Each steps through its nested-loop code with the executing line highlighted, the cells being read (yellow) and written (green), the loop variables, and the einsum string ('ij,jk->ik').
- **Tensors in a neural network:** an image classifier's 4D activations [batch, height, width, channels] and weight tensors, with live element counts, memory size, parameter counts and multiply-adds as you change the batch size, image size, channels, filters and classes.

---

## How it works

| Topic | Method |
|---|---|
| Expression parsing & derivatives | math.js parse + symbolic `derivative`, with a numerical fallback |
| Curve drawing | Adaptive sampling; asymptotes are detected by bisection, so `tan x` and `1/x` break cleanly |
| Fluid | *Stable Fluids* (J. Stam, SIGGRAPH 1999): semi-Lagrangian advection, implicit diffusion, Gauss–Seidel pressure projection. Dye uses MacCormack advection with a limiter to stay sharp |
| Light | FDTD on a Yee grid (K. Yee, 1966), 2D TMz polarisation, Courant number 0.5, graded absorbing border, PEC metal, dielectric glass |
| Control | 960 Hz physics with semi-implicit Euler; PID with a filtered derivative, conditional-integration anti-windup and optional derivative-on-measurement; full nonlinear cart-pole dynamics |
| Fourier | Direct DFT of 512 samples per period; complex DFT of a 256-point arc-length-resampled path for the drawings; 3D views use a small orthographic camera (yaw/pitch, drag to rotate, animated flights between named views) drawn on Canvas 2D |
| Laplace | Exact impulse and step responses from partial-fraction residues, h(t) = Σ rₖ e^{pₖt}; gain normalized to unit DC gain |
| Circuits | 4th-order Runge–Kutta on (q, i), 20 sub-steps per frame; resonance curve from the impedance, \|I\| = V/\|Z\| |
| Learning | Full-batch gradient descent / momentum / Adam; hand-written backpropagation for a tanh network with cross-entropy loss; landscape slice along two random directions scaled to the weight norm, with the path projected by least squares; Lagrange: dense sampling of the constraint path, projected-gradient sliding with step halving, λ = ∇f·∇g / ‖∇g‖² |
| Linear algebra | Closed-form 2×2 eigen-decomposition from the trace and determinant (real, repeated and complex cases); power iteration with normalization; sample covariance and its eigenvectors for PCA |
| Probability | Exact Bayes posterior with whole-number frequency counts; Monte-Carlo sampling for the CLT; Markov stationary distribution by power iteration, with the mixing rate |λ₂| estimated from the decay of the worst-case distance to π |
| Convolution | Riemann-sum convolution on a 501-point grid with exact handling of impulse (spike) signals and kernels; circular convolution two ways (direct and FFT) on 256 points; true 2D convolution (flipped kernel, clamped edges) |
| Images | Radix-2 2D FFT (256²); separable convolution; median by window sort; bilateral with a range-weight lookup table; ROF total variation via Chambolle's dual projection (τ = 1/8); periodic-noise detection by z-score plus local peak contrast |

**Code layout**

```
index.html, styles.css, app.js     graph plotter
sim.html, sim.css, sim.js          simulation shell (tabs, canvas, input routing, loop)
sim-ui.js                          shared controls and color maps
sim-fluid.js                       Navier–Stokes
sim-maxwell.js                     Maxwell FDTD + 3D plane wave & polarization
sim-control.js                     PID and the three plants
sim-fourier.js                     Fourier series, winding machine, drawing with circles, 3D time↔frequency and complex helix
sim-laplace.js                     s-plane, responses, 3D landscape
sim-circuit.js                     RC / RL / LC / series & parallel RLC circuits and the mechanical twin
sim-learning.js                    line fitting, optimizer race, tiny neural network + loss-landscape slice, Lagrange multipliers
sim-image.js                       image noise models, median / FFT mask / box / Gaussian / bilateral / total-variation denoising
sim-linalg.js                      matrices as transformations, eigenvectors & power iteration, PCA
sim-prob.js                        Bayes' rule, central limit theorem, Markov chains & PageRank
sim-conv.js                        convolution: slide & sum, convolution theorem, 2D image kernels
sim-tensor.js                      tensors: shapes & strides in memory, operations as loops (einsum), neural-net shapes
```

Each simulation is a self-contained module with the same small interface: `mount`, `resize`, `frame`, `pointer`. Adding a new tab means adding one file and one button.

---

## Verification

The numerics were checked against known answers, not only by eye:

- **Fourier:** square-wave coefficients match 4/π, 0, 4/(3π), … to four decimals; the winding machine finds the correct peaks and heights.
- **Laplace:** impulse responses match closed-form solutions (first-order, underdamped, overdamped, unstable) to machine precision; a zero on the jω axis gives exactly zero gain there.
- **Circuits:** the RC, RL, LC, underdamped series RLC and parallel RLC responses match their closed-form solutions to about 10⁻¹² relative error; LC energy is conserved to 10⁻¹³ over 10 s. Driving at f₀ gives exactly V/R (series current) and I·R (parallel voltage).
- **Learning:** gradient descent converges to the exact least-squares line, and diverges right at the theoretical limit (η = 0.98 here); backpropagation matches finite-difference gradients to 6×10⁻⁸; the network reaches 100% accuracy on circle, XOR and moons and 92% on the spiral, and one neuron fails on the circle as expected.
- **Lagrange:** the fence gives exactly x = y = P/4 with λ = P/8; on every problem λ* matches the measured df*/dc to 3 or 4 digits; the lasso optimum lands exactly on a corner (w₂ = 0) while ridge keeps both weights non-zero.
- **Images:** the FFT matches a direct DFT to 10⁻¹²; each filter wins on its own noise type (median +13 dB on salt & pepper vs. +6 dB for blur; notch +11 dB on stripes vs. +2 dB for blur; total variation +12 dB on grain vs. about +4 dB for blur). Auto-notch finds exactly the stripe at any angle and nothing on clean images, and the total-variation energy decreases at every iteration.
- **Linear algebra:** on 10 test matrices Av = λv holds to machine precision, the eigenvalues add up to the trace and multiply to the determinant, and det equals the signed area of the transformed square; power iteration reaches the dominant eigenvector; a brute-force search over 18,000 angles finds the PCA maximum exactly at PC1 with value λ₁.
- **3D light wave:** E · B = 0 and |B| = |E|/c exactly at every point and time, E × B never points backward, circular light keeps |E| constant, and the simulated time-averaged intensity behind the polarizer matches Malus's cos²θ (1.00 / 0.75 / 0.25 / 0.00 at 0° / 30° / 60° / 90°; circular light loses exactly half).
- **Probability:** Bayes gives 16.10% for the classic 1% / 95% / 5% case and 78.48% after a second positive test, both matching hand calculation; CLT averages match μ and σ/√n for every distribution, with skewness falling 1.96 → 0.89 → 0.36 as n goes 1 → 5 → 30 (exponential); Markov π satisfies πP = π to 10⁻¹⁵ and matches 400,000 simulated hops; the estimated |λ₂| = 0.4732 for the weather chain equals the exact eigenvalue.
- **Convolution:** step * RC response matches 1 − e^(−t/τ) within the 0.02 grid spacing; echo and edge-detector outputs are exact away from the jumps; ∫(f*g) = ∫f·∫g; direct and FFT convolution agree to about 10⁻¹⁵ for every signal/kernel pair; in 2D the identity kernel changes nothing, blur preserves flat areas, Sobel ↔ responds only to vertical edges, and the kernel flip was checked with a one-pixel shift.
- **Tensors:** strides for [2,3,4] are [12,4,1] and contiguous reading visits memory 0…23 in order; transposing gives T[k][j][i] = X[i][j][k] for every cell with no data moved; reshape is copy-free when contiguous and copies correctly when not; all five operation loops produce the right results; the neural-net parameter and multiply-add counts match hand calculation.
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

**Designed and written by Claude** (Claude Opus 5.5, an AI model by [Anthropic](https://www.anthropic.com)), working in Claude Code at the request of, and with direction from, the project owner. The owner set the goals and topics: making the math of CS and engineering visual, then adding Maxwell, Navier–Stokes, PID control, Fourier, Laplace, RLC circuits, machine learning, Lagrange multipliers, image denoising, linear algebra, probability, convolution and tensors.

Claude chose the technology (plain JavaScript and Canvas, no build step), designed the interfaces and explanations, implemented every solver and visualization, and verified the numerics described above.

**Third-party libraries:** [math.js](https://mathjs.org) (Apache-2.0) and [KaTeX](https://katex.org) (MIT), both loaded from CDNs.

**Methods credited to their originators:** Jos Stam (*Stable Fluids*, 1999), Kane S. Yee (FDTD, 1966), and the classical work of Fourier, Laplace, Maxwell, Navier and Stokes.
