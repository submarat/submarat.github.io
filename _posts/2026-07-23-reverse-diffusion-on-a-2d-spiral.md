---
layout: post
title: "Reverse diffusion on a 2D spiral"
date: 2026-07-23
excerpt: "A minimal from-scratch DDPM trained on a 2D spiral, animated to show random noise being denoised back onto the data manifold step by step — plus a side-by-side comparison of stochastic vs. deterministic sampling."
reading_time_minutes: 4
---

Diffusion models are usually explained through images, where the actual denoising trajectory is invisible inside a high-dimensional latent space — you see the final picture, not the path it took to get there. If the data is 2D instead, every step of the reverse process can just be plotted directly. So: a tiny DDPM, trained from scratch on points sampled along a spiral, animated with [Manim](https://www.manim.community/) to show exactly what reverse diffusion is doing. Code's [on GitHub](https://github.com/submarat/spiral-diffusion); built in conversation with Claude.

The setup is deliberately minimal — a time-conditioned MLP with a couple hundred parameters, trained in seconds, predicting the noise added to a spiral point at a random timestep. Sampling starts from pure Gaussian noise and repeatedly asks the model to denoise, 200 steps, until the points land on the spiral.

<video src="/videos/spiral-diffusion/ReverseDiffusion.mp4" controls width="100%"></video>

Each dot is colored by where it ends up along the spiral, and the fading trail behind it is its actual path through the reverse process — not smoothed or idealized, just the literal sequence of positions the sampler visited. The jaggedness is real: standard DDPM sampling injects fresh Gaussian noise at every single step, so even a model that's fully converged produces a noisy random walk on the way to a clean sample.

That raised an obvious follow-up: how much of that jaggedness is necessary? DDIM answers this — the same trained model, but with the injected-noise term set to zero, turns sampling into a deterministic function of the starting noise:

<video src="/videos/spiral-diffusion/ReverseDiffusionDeterministic.mp4" controls width="100%"></video>

Same model, same starting positions, and the destinations end up nearly identical — but the paths there are smooth, direct curves instead of noisy walks. It's a clean, visual demonstration of something that's easy to state abstractly (stochastic ancestral sampling vs. deterministic ODE-like sampling) but nicer to actually watch: the denoising *direction* the model has learned is the same either way, only whether noise gets re-injected at each step differs.
