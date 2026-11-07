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

## Files

- `index.html`: page layout
- `styles.css`: styling, light and dark mode
- `app.js`: parsing (math.js), sampling, canvas rendering, animation, interaction
