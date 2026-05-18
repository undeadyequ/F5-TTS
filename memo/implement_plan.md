from f5_tts.scripts.count_params_gflops import transformer

### Sample function

```python
trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
self.transformer.clear_cache()
```


### sway sampling

```python
if sway_sampling_coef is not None:
    t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
```

### flow-grpo

```python

def sed_sample(cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=65536,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,):
    ...
    latents = y0
    for i, t in enumerate(timesteps):
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
    
            # predict flow (cond)
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
                return pred
    
            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                cfg_infer=True,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength
        
        noise_pred = fn(t, latents)
        latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
            self.scheduler, 
            noise_pred,
            t.unsqueeze(0),
            latents.float(),
            noise_level=noise_level,)
    sample = latents.to(dtype=noise_pred.dtype)
    return sample
```