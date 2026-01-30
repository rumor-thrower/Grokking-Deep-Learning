
# Grokking Deep Learning — `Julia` Port & Learning

This repository organizes materials from the original book "Grokking Deep Learning" (mostly notebooks) for learning purposes and aims to port parts of the examples to `Julia`.

[Grokking Deep Learning — Official page](https://manning.com/books/grokking-deep-learning?a_aid=grokkingdl&a_bid=32715258)

Note: This repository may include the following submodules or references:
- `Book`: The original repository containing the source (mostly `Python`) example notebooks.
- `BookJuliaPort`: Community/third-party examples ported to `Julia`.


## Primary Goals
- Use the original book's example notebooks (originally in `Python`) as learning material.
- Port examples step-by-step to `Julia` for educational comparison and experiments.
- Document design and numerical differences encountered during porting.

## Quick Start (Recommended)

Prerequisites: `Julia` 1.8 or later, `git`, and optionally `jupyter` (or `jupyterlab`).

1) Activate the `Julia` environment at the repository root and install packages:

```bash
julia --project=. -e 'import Pkg; Pkg.activate("."); Pkg.instantiate()'
```

If the repository contains submodules (on first clone), initialize them first:

```bash
git submodule update --init --recursive
```

2) To open the notebooks in Jupyter (requires the `IJulia` kernel):

```bash
julia --project=. -e 'import Pkg; Pkg.add("IJulia"); using IJulia; installkernel("Grok")'
jupyter lab
```

Or to run Pluto notebooks (if you prefer Pluto):

```bash
julia --project=. -e 'import Pkg; Pkg.add("Pluto"); using Pluto; Pluto.run()'
```


## Repository Structure (Summary)
- `Book/`: Original (mostly `Python`) notebooks and data files used as source material — usually a submodule of the original repo.
- `BookJuliaPort/`: Community / third-party `Julia` ports of examples (submodule).
- `src/Grok.jl`: Supporting `Julia` code (utilities used during porting).
- `Project.toml`, `Manifest.toml`: `Julia` environment and dependencies.

Example data files: `shakespear.txt`, `labels.txt`, `spam.txt`, etc., are used by some notebooks.


## Porting Guide (Short)
- Follow the original notebook cell-by-cell and reproduce the same numbers/outputs.
- Replace NumPy/Matplotlib calls with `Julia` equivalents such as `LinearAlgebra`, `Plots.jl`, or `JuliaImages.jl`.
- For automatic differentiation examples, compare `Zygote.jl` with simple numeric-differentiation implementations.
- Record major numerical or performance differences as comments in the notebook when porting.

## Porting Checklist (example)
- Confirm whether original cell inputs/outputs are reproduced.
- Add required packages to `Project.toml`.
- Fix plot labels/axes so visualizations match the originals.


## Contributing
- To port or modify, [fork](https://github.com/rumor-thrower/Grokking-Deep-Learning/fork) → branch → [open a PR](https://github.com/rumor-thrower/Grokking-Deep-Learning/pull/new/develop). Include: changed files, short run instructions, and (optionally) screenshots of reproduced examples.

## License & Copyright
- This repository is a learning-oriented port/notes collection. Respect the original author's copyright and licensing when using source materials — check the original work's license/permissions.

## Contact / Further Work
- Plans include porting more chapters to `Julia` or adding simple automated checks that validate cell reproduction.
- For questions or contributions, please use this repository's [issue tracker](https://github.com/rumor-thrower/Grokking-Deep-Learning/issues).

---

Refer to the `Book/` directory for the list and links of original (`Python`) notebooks. `Julia` port examples will be added sequentially to `src/`.
