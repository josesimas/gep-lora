# Base models and LoRAs: speed comparison

Made 2026-09-15 15:04 by `python -m adapters.base_models_and_loras_comparison --loader both`, in 31m 54s.

|  |  |
|---|---:|
| GPU | NVIDIA RTX A6000 (51.53 GB) |
| Python / libraries | 3.11.9 / torch 2.11.0+cu128 / transformers 5.5.0 / peft 0.20.0 / unsloth 2026.8.15 |
| fixed-length test | 64 new tokens, median of 3, batch 1 and 8 |
| natural test | 8 prompts, generate() calls of 8, up to 250 new tokens |
| base models | `unsloth/Qwen3.5-0.8B`, `unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit`, `unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit` |

## Summary

| base model (loader) | time to first answer | LoRA attach (mean) | decode tok/s, base (b=1) | decode tok/s, with LoRA (mean) | LoRA effect | noise | VRAM after load |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B (unsloth) | 28.36 s | 0.21 s | 20.4 | 13.1 | -36% | +2% | 0.87 GB |
| Qwen3.5-0.8B (plain) | 18.14 s | 0.21 s | 16.1 | 11.2 | -31% | +2% | 0.83 GB |
| qwen2.5-0.5b-instruct-unsloth-bnb-4bit (unsloth) | 19.59 s | 0.33 s | 26.2 | 19.7 | -25% | +1% | 0.57 GB |
| qwen2.5-0.5b-instruct-unsloth-bnb-4bit (plain) | 10.40 s | 0.35 s | 23.6 | 11.8 | -50% | -1% | 0.54 GB |
| qwen2.5-1.5b-instruct-unsloth-bnb-4bit (unsloth) | 20.44 s | 0.45 s | 23.2 | 16.7 | -28% | +1% | 1.58 GB |
| qwen2.5-1.5b-instruct-unsloth-bnb-4bit (plain) | 10.72 s | 0.50 s | 20.8 | 10.6 | -49% | +0% | 1.53 GB |

*Time to first answer* is import + model load + inference setup + the first generate(); add *LoRA attach* for the same with an adapter on. *LoRA effect* is the mean over this base's adapters of the change in decode speed at a fixed answer length -- negative is slower. *noise* is the same change for the bare base measured a second time, at the end: a LoRA effect no bigger than it is not one.

## 1. Loading a base model, cold

| base model (loader) | import | model_load | inference_setup | first generate (8 tok) | time to first answer | VRAM after load | whole worker | GPU before |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B (unsloth) | 9.54 s | 13.77 s | 0.00 s | 5.04 s | **28.36 s** | 0.87 GB | 358.36 s | 0%, 1.26 GB |
| Qwen3.5-0.8B (plain) | 4.95 s | 12.06 s | 0.00 s | 1.13 s | **18.14 s** | 0.83 GB | 413.56 s | 0%, 1.33 GB |
| qwen2.5-0.5b-instruct-unsloth-bnb-4bit (unsloth) | 9.55 s | 9.16 s | 0.00 s | 0.87 s | **19.59 s** | 0.57 GB | 264.38 s | 0%, 1.29 GB |
| qwen2.5-0.5b-instruct-unsloth-bnb-4bit (plain) | 4.91 s | 4.79 s | 0.00 s | 0.70 s | **10.40 s** | 0.54 GB | 308.28 s | 1%, 1.28 GB |
| qwen2.5-1.5b-instruct-unsloth-bnb-4bit (unsloth) | 9.54 s | 9.82 s | 0.00 s | 1.07 s | **20.44 s** | 1.58 GB | 266.46 s | 1%, 1.22 GB |
| qwen2.5-1.5b-instruct-unsloth-bnb-4bit (plain) | 4.87 s | 5.11 s | 0.00 s | 0.74 s | **10.72 s** | 1.53 GB | 290.38 s | 0%, 1.22 GB |

Each row is a fresh interpreter, so nothing was imported or loaded before it. *whole worker* is that process from launch to exit, every measurement below included; the gap between it and the phases before it is also the interpreter starting up, which no phase covers. *GPU before* is the card's utilization and memory in use just before the worker started, by anything: on an idle card it is near 0%, and anything above 10% means the row was measured while sharing the GPU.

## 2. Attaching a LoRA

| slot | Qwen3.5-0.8B (unsloth) | Qwen3.5-0.8B (plain) | qwen2.5-0.5b-instruct-unsloth-bnb-4bit (unsloth) | qwen2.5-0.5b-instruct-unsloth-bnb-4bit (plain) | qwen2.5-1.5b-instruct-unsloth-bnb-4bit (unsloth) | qwen2.5-1.5b-instruct-unsloth-bnb-4bit (plain) |
|---|---:|---:|---:|---:|---:|---:|
| Lora001 | r16 · 0.22 s | r16 · 0.23 s | r16 · 0.34 s | r16 · 0.48 s | r4 · 0.33 s | r4 · 0.50 s |
| Lora002 | r16 · 0.24 s | r16 · 0.23 s | r16 · 0.35 s | r16 · 0.34 s | r8 · 0.38 s | r8 · 0.38 s |
| Lora003 | r8 · 0.15 s | r8 · 0.14 s | r8 · 0.28 s | r8 · 0.29 s | r16 · 0.48 s | r16 · 0.50 s |
| Lora004 | r4 · 0.13 s | r4 · 0.13 s | r4 · 0.20 s | r4 · 0.20 s | r32 · 0.68 s | r32 · 0.74 s |
| Lora005 | r32 · 0.32 s | r32 · 0.29 s | r32 · 0.46 s | r32 · 0.43 s | r8 · 0.38 s | r8 · 0.39 s |

Each cell is the adapter's rank and how long `PeftModel.from_pretrained()` took to read it and wrap the base with it. Every adapter was attached to the bare base on its own, and taken off again before the next.

## 3. Inference, with and without a LoRA

### Qwen3.5-0.8B (unsloth)

|  | rank | prefill b=1 | decode tok/s b=1 | vs base | prefill b=8 | decode tok/s b=8 | vs base | natural: seconds | tokens | vs base | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **base** |  | 81 ms | 20.4 |  | 105 ms | 163.1 |  | 12.30 s | 1801 |  | 1.03 GB |
| Lora001 | r16 | 126 ms | 13.4 | -34% | 128 ms | 115.5 | -29% | 17.31 s | 2000 | +41% | 1.03 GB |
| Lora002 | r16 | 126 ms | 13.3 | -35% | 130 ms | 115.4 | -29% | 17.34 s | 2000 | +41% | 1.03 GB |
| Lora003 | r8 | 129 ms | 13.0 | -36% | 130 ms | 115.1 | -29% | 17.31 s | 1772 | +41% | 1.02 GB |
| Lora004 | r4 | 131 ms | 12.8 | -37% | 126 ms | 113.0 | -31% | 17.60 s | 2000 | +43% | 1.01 GB |
| Lora005 | r32 | 124 ms | 13.2 | -35% | 126 ms | 116.2 | -29% | 17.43 s | 2000 | +42% | 1.06 GB |
| base, again |  | 99 ms | 20.7 | +2% | 102 ms | 161.7 | -1% | 12.26 s | 1801 | -0% | 1.01 GB |

### Qwen3.5-0.8B (plain)

|  | rank | prefill b=1 | decode tok/s b=1 | vs base | prefill b=8 | decode tok/s b=8 | vs base | natural: seconds | tokens | vs base | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **base** |  | 296 ms | 16.1 |  | 315 ms | 132.8 |  | 15.09 s | 1819 |  | 1.01 GB |
| Lora001 | r16 | 318 ms | 11.2 | -30% | 346 ms | 98.0 | -26% | 20.41 s | 1611 | +35% | 1.03 GB |
| Lora002 | r16 | 320 ms | 11.1 | -31% | 350 ms | 98.0 | -26% | 20.51 s | 1715 | +36% | 1.03 GB |
| Lora003 | r8 | 321 ms | 11.1 | -31% | 347 ms | 96.8 | -27% | 20.55 s | 1697 | +36% | 1.02 GB |
| Lora004 | r4 | 319 ms | 10.9 | -32% | 346 ms | 97.8 | -26% | 20.31 s | 1716 | +35% | 1.01 GB |
| Lora005 | r32 | 318 ms | 11.5 | -29% | 341 ms | 98.2 | -26% | 20.44 s | 1708 | +35% | 1.06 GB |
| base, again |  | 294 ms | 16.4 | +2% | 317 ms | 135.3 | +2% | 14.96 s | 1819 | -1% | 1.01 GB |

### qwen2.5-0.5b-instruct-unsloth-bnb-4bit (unsloth)

|  | rank | prefill b=1 | decode tok/s b=1 | vs base | prefill b=8 | decode tok/s b=8 | vs base | natural: seconds | tokens | vs base | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **base** |  | 49 ms | 26.2 |  | 56 ms | 152.0 |  | 13.04 s | 1497 |  | 0.68 GB |
| Lora001 | r16 | 98 ms | 19.8 | -24% | 104 ms | 116.1 | -24% | 9.20 s | 282 | -29% | 0.72 GB |
| Lora002 | r16 | 97 ms | 19.7 | -25% | 113 ms | 116.7 | -23% | 9.19 s | 282 | -30% | 0.72 GB |
| Lora003 | r8 | 99 ms | 19.9 | -24% | 106 ms | 113.1 | -26% | 5.26 s | 295 | -60% | 0.70 GB |
| Lora004 | r4 | 97 ms | 19.2 | -27% | 105 ms | 114.6 | -25% | 5.95 s | 269 | -54% | 0.68 GB |
| Lora005 | r32 | 99 ms | 19.8 | -24% | 107 ms | 116.4 | -23% | 16.56 s | 488 | +27% | 0.78 GB |
| base, again |  | 50 ms | 26.4 | +1% | 59 ms | 153.1 | +1% | 13.02 s | 1497 | -0% | 0.68 GB |

### qwen2.5-0.5b-instruct-unsloth-bnb-4bit (plain)

|  | rank | prefill b=1 | decode tok/s b=1 | vs base | prefill b=8 | decode tok/s b=8 | vs base | natural: seconds | tokens | vs base | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **base** |  | 58 ms | 23.6 |  | 55 ms | 205.1 |  | 9.67 s | 1481 |  | 0.59 GB |
| Lora001 | r16 | 98 ms | 11.9 | -50% | 96 ms | 102.9 | -50% | 11.18 s | 329 | +16% | 0.62 GB |
| Lora002 | r16 | 96 ms | 11.9 | -50% | 96 ms | 102.9 | -50% | 11.19 s | 329 | +16% | 0.62 GB |
| Lora003 | r8 | 96 ms | 11.7 | -51% | 97 ms | 101.6 | -50% | 19.68 s | 495 | +104% | 0.61 GB |
| Lora004 | r4 | 97 ms | 11.8 | -50% | 94 ms | 102.8 | -50% | 6.44 s | 267 | -33% | 0.59 GB |
| Lora005 | r32 | 98 ms | 11.9 | -50% | 97 ms | 102.9 | -50% | 18.66 s | 492 | +93% | 0.66 GB |
| base, again |  | 57 ms | 23.3 | -1% | 58 ms | 205.2 | +0% | 9.73 s | 1481 | +1% | 0.59 GB |

### qwen2.5-1.5b-instruct-unsloth-bnb-4bit (unsloth)

|  | rank | prefill b=1 | decode tok/s b=1 | vs base | prefill b=8 | decode tok/s b=8 | vs base | natural: seconds | tokens | vs base | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **base** |  | 61 ms | 23.2 |  | 67 ms | 133.8 |  | 14.90 s | 1576 |  | 1.80 GB |
| Lora001 | r4 | 115 ms | 16.9 | -27% | 120 ms | 98.0 | -27% | 1.60 s | 109 | -89% | 1.81 GB |
| Lora002 | r8 | 117 ms | 16.9 | -27% | 123 ms | 98.0 | -27% | 5.04 s | 172 | -66% | 1.84 GB |
| Lora003 | r16 | 120 ms | 16.6 | -28% | 136 ms | 98.0 | -27% | 3.01 s | 123 | -80% | 1.89 GB |
| Lora004 | r32 | 120 ms | 16.4 | -29% | 124 ms | 97.9 | -27% | 2.85 s | 122 | -81% | 2.01 GB |
| Lora005 | r8 | 116 ms | 16.8 | -28% | 120 ms | 99.3 | -26% | 1.42 s | 95 | -90% | 1.84 GB |
| base, again |  | 57 ms | 23.3 | +1% | 64 ms | 134.1 | +0% | 14.94 s | 1576 | +0% | 1.80 GB |

### qwen2.5-1.5b-instruct-unsloth-bnb-4bit (plain)

|  | rank | prefill b=1 | decode tok/s b=1 | vs base | prefill b=8 | decode tok/s b=8 | vs base | natural: seconds | tokens | vs base | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **base** |  | 67 ms | 20.8 |  | 62 ms | 176.9 |  | 11.33 s | 1372 |  | 1.64 GB |
| Lora001 | r4 | 111 ms | 10.5 | -49% | 117 ms | 89.3 | -50% | 1.73 s | 109 | -85% | 1.61 GB |
| Lora002 | r8 | 116 ms | 10.6 | -49% | 111 ms | 89.5 | -49% | 5.61 s | 187 | -50% | 1.63 GB |
| Lora003 | r16 | 110 ms | 10.6 | -49% | 111 ms | 89.2 | -50% | 3.60 s | 147 | -68% | 1.67 GB |
| Lora004 | r32 | 114 ms | 10.5 | -49% | 115 ms | 87.3 | -51% | 3.23 s | 122 | -71% | 1.74 GB |
| Lora005 | r8 | 112 ms | 10.6 | -49% | 113 ms | 88.6 | -50% | 1.55 s | 95 | -86% | 1.63 GB |
| base, again |  | 65 ms | 20.9 | +0% | 69 ms | 175.9 | -1% | 11.34 s | 1372 | +0% | 1.64 GB |

*base, again* is the bare base measured a second time, after every adapter had been on and off: how far it is from the first *base* row is the noise in these numbers, and an adapter's *vs base* within that distance is not a measurable effect.

*prefill* is a generate() of one new token: reading the prompt. *decode tok/s* is the tokens after that one, divided by the time after it, summed over the batch -- so b=8 is throughput across eight answers at once. Both are at a fixed answer length, the same work with and without the adapter. *natural* is the 8 prompts answered in generate() calls of 8, stopping where the model stops (at most 250 tokens), the way a generated script answers: an adapter that changes how long the answers are shows up there and not in the decode columns.

## 4. LoRA effect on decode speed, by rank

| rank | Qwen3.5-0.8B (unsloth) b=1 | Qwen3.5-0.8B (unsloth) b=8 | Qwen3.5-0.8B (plain) b=1 | Qwen3.5-0.8B (plain) b=8 | qwen2.5-0.5b-instruct-unsloth-bnb-4bit (unsloth) b=1 | qwen2.5-0.5b-instruct-unsloth-bnb-4bit (unsloth) b=8 | qwen2.5-0.5b-instruct-unsloth-bnb-4bit (plain) b=1 | qwen2.5-0.5b-instruct-unsloth-bnb-4bit (plain) b=8 | qwen2.5-1.5b-instruct-unsloth-bnb-4bit (unsloth) b=1 | qwen2.5-1.5b-instruct-unsloth-bnb-4bit (unsloth) b=8 | qwen2.5-1.5b-instruct-unsloth-bnb-4bit (plain) b=1 | qwen2.5-1.5b-instruct-unsloth-bnb-4bit (plain) b=8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r4 | -37% | -31% | -32% | -26% | -27% | -25% | -50% | -50% | -27% | -27% | -49% | -50% |
| r8 | -36% | -29% | -31% | -27% | -24% | -26% | -51% | -50% | -27% | -26% | -49% | -50% |
| r16 | -35% | -29% | -31% | -26% | -24% | -23% | -50% | -50% | -28% | -27% | -49% | -50% |
| r32 | -35% | -29% | -29% | -26% | -24% | -23% | -50% | -50% | -29% | -27% | -49% | -51% |

The change in decode speed against the bare base, at a fixed answer length. Where two slots share a rank, the two are averaged.

## Prompts

1. What are the early signs of dehydration?
2. How is high blood pressure usually treated?
3. What should I do if I think I have the flu?
4. Why is it important to finish a course of antibiotics?
5. Help me plan my week.
6. What's the capital of France?
7. Tell me about the ocean.
8. Explain how a bicycle stays upright.

## Raw results

<details><summary>JSON, one entry per worker</summary>

```json
[
 {
  "base_model": "unsloth/Qwen3.5-0.8B",
  "loader": "unsloth",
  "ok": true,
  "configs": [
   {
    "resting_vram": 865860608,
    "first_generate": 5.037570299999061,
    "fixed": {
     "1": {
      "warmup": 0.43140779999885126,
      "prefill": 0.08067129999835743,
      "total": 3.170200799999293,
      "tokens": 64,
      "forced": true,
      "decode_tps": 20.391454426954304
     },
     "8": {
      "warmup": 4.441256599999178,
      "prefill": 0.10479750000013155,
      "total": 3.194114300000365,
      "tokens": 512,
      "forced": true,
      "decode_tps": 163.14286705719593
     }
    },
    "natural": {
     "seconds": 12.300169599999208,
     "tokens": 1801,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1028208128,
    "name": "base",
    "slot": null,
    "rank": null
   },
   {
    "name": "0.8b_modellora_adapter",
    "slot": "Lora001",
    "rank": 16,
    "attach": 0.22393509999892558,
    "resting_vram": 891420160,
    "first_generate": 1.818785499999649,
    "fixed": {
     "1": {
      "warmup": 0.7387525000012829,
      "prefill": 0.12571979999847827,
      "total": 4.839205200001743,
      "tokens": 64,
      "forced": true,
      "decode_tps": 13.36590540833252
     },
     "8": {
      "warmup": 1.2782329000001482,
      "prefill": 0.12847540000075242,
      "total": 4.491342299999815,
      "tokens": 512,
      "forced": true,
      "decode_tps": 115.52037033266092
     }
    },
    "natural": {
     "seconds": 17.307578699997975,
     "tokens": 2000,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1032574464,
    "detach": 0.03546180000193999
   },
   {
    "name": "0.8b_modellora_adapter",
    "slot": "Lora002",
    "rank": 16,
    "attach": 0.23833340000055614,
    "resting_vram": 892468736,
    "first_generate": 0.743560000002617,
    "fixed": {
     "1": {
      "warmup": 0.7285875999987184,
      "prefill": 0.12623639999947045,
      "total": 4.873948799999198,
      "tokens": 64,
      "forced": true,
      "decode_tps": 13.269548509299682
     },
     "8": {
      "warmup": 0.7002076999997371,
      "prefill": 0.12966899999810266,
      "total": 4.498888900001475,
      "tokens": 512,
      "forced": true,
      "decode_tps": 115.35239963536992
     }
    },
    "natural": {
     "seconds": 17.34340169999996,
     "tokens": 2000,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1032574464,
    "detach": 0.034329000001889654
   },
   {
    "name": "0.8b_modellora_adapter",
    "slot": "Lora003",
    "rank": 8,
    "attach": 0.14816760000030627,
    "resting_vram": 879689216,
    "first_generate": 2.1948637000023155,
    "fixed": {
     "1": {
      "warmup": 0.7002920999984781,
      "prefill": 0.12878499999715132,
      "total": 4.958976100002474,
      "tokens": 64,
      "forced": true,
      "decode_tps": 13.0429622132198
     },
     "8": {
      "warmup": 1.9534736999994493,
      "prefill": 0.12967579999894951,
      "total": 4.5084105000023555,
      "tokens": 512,
      "forced": true,
      "decode_tps": 115.10174388953227
     }
    },
    "natural": {
     "seconds": 17.30882399999973,
     "tokens": 1772,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1019794944,
    "detach": 0.037118399999599205
   },
   {
    "name": "0.8b_modellora_adapter",
    "slot": "Lora004",
    "rank": 4,
    "attach": 0.1286433000022953,
    "resting_vram": 873299456,
    "first_generate": 1.2208343999991484,
    "fixed": {
     "1": {
      "warmup": 0.7278026999993017,
      "prefill": 0.1306789000009303,
      "total": 5.061615700000402,
      "tokens": 64,
      "forced": true,
      "decode_tps": 12.776476875551669
     },
     "8": {
      "warmup": 1.3709110999989207,
      "prefill": 0.1264415999976336,
      "total": 4.58630290000292,
      "tokens": 512,
      "forced": true,
      "decode_tps": 113.0079987015297
     }
    },
    "natural": {
     "seconds": 17.60412890000225,
     "tokens": 2000,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1013405184,
    "detach": 0.03437339999800315
   },
   {
    "name": "0.8b_modellora_adapter",
    "slot": "Lora005",
    "rank": 32,
    "attach": 0.31814870000016526,
    "resting_vram": 918027776,
    "first_generate": 2.0154922000001534,
    "fixed": {
     "1": {
      "warmup": 0.7169030999975803,
      "prefill": 0.12435040000127628,
      "total": 4.8825594000009005,
      "tokens": 64,
      "forced": true,
      "decode_tps": 13.240275910537973
     },
     "8": {
      "warmup": 1.8450703000016802,
      "prefill": 0.12642219999906956,
      "total": 4.462124899997434,
      "tokens": 512,
      "forced": true,
      "decode_tps": 116.24413269853353
     }
    },
    "natural": {
     "seconds": 17.426567900001828,
     "tokens": 2000,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1058133504,
    "detach": 0.036622499999793945
   },
   {
    "resting_vram": 866909696,
    "first_generate": 0.47734549999950104,
    "fixed": {
     "1": {
      "warmup": 0.5726538999988406,
      "prefill": 0.09930499999973108,
      "total": 3.139332499999,
      "tokens": 64,
      "forced": true,
      "decode_tps": 20.723496744689037
     },
     "8": {
      "warmup": 0.4716740999974718,
      "prefill": 0.1022064999997383,
      "total": 3.219121799997083,
      "tokens": 512,
      "forced": true,
      "decode_tps": 161.69833039750208
     }
    },
    "natural": {
     "seconds": 12.25834549999854,
     "tokens": 1801,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1007015424,
    "name": "base_again",
    "slot": null,
    "rank": null
   }
  ],
  "environment": {
   "python": "3.11.9",
   "torch": "2.11.0+cu128",
   "transformers": "5.5.0",
   "peft": "0.20.0",
   "unsloth": "2026.8.15",
   "gpu": "NVIDIA RTX A6000",
   "gpu_memory": 51526500352
  },
  "cold": {
   "import": 9.54296720000275,
   "model_load": 13.774961999999505,
   "inference_setup": 0.003166800001054071,
   "vram_after_load": 865860608,
   "first_generate": 5.037570299999061
  },
  "process_seconds": 358.36105159999715,
  "gpu_before": {
   "utilization": 0.0,
   "memory_used": 1264582656.0
  }
 },
 {
  "base_model": "unsloth/Qwen3.5-0.8B",
  "loader": "plain",
  "ok": true,
  "configs": [
   {
    "resting_vram": 832196096,
    "first_generate": 1.1290845000003173,
    "fixed": {
     "1": {
      "warmup": 0.8079132999991998,
      "prefill": 0.29626790000111214,
      "total": 4.208099999999831,
      "tokens": 64,
      "forced": true,
      "decode_tps": 16.10498569200366
     },
     "8": {
      "warmup": 0.8380432000012661,
      "prefill": 0.3145651999984693,
      "total": 4.109663299997919,
      "tokens": 512,
      "forced": true,
      "decode_tps": 132.80289118219974
     }
    },
    "natural": {
     "seconds": 15.088602299998456,
     "tokens": 1819,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1008033280,
    "name": "base",
    "slot": null,
    "rank": null
   },
   {
    "name": "0.8b_modellora_adapter",
    "slot": "Lora001",
    "rank": 16,
    "attach": 0.23307940000086091,
    "resting_vram": 866275328,
    "first_generate": 0.9509881999983918,
    "fixed": {
     "1": {
      "warmup": 0.9591942999977618,
      "prefill": 0.3176729999977397,
      "total": 5.9408573000000615,
      "tokens": 64,
      "forced": true,
      "decode_tps": 11.203616427790564
     },
     "8": {
      "warmup": 1.0094858999982534,
      "prefill": 0.34556959999827086,
      "total": 5.4879184999990684,
      "tokens": 512,
      "forced": true,
      "decode_tps": 98.0096858071847
     }
    },
    "natural": {
     "seconds": 20.407934099999693,
     "tokens": 1611,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1033330176,
    "detach": 0.050227399999130284
   },
   {
    "name": "0.8b_modellora_adapter",
    "slot": "Lora002",
    "rank": 16,
    "attach": 0.22878209999907995,
    "resting_vram": 866275328,
    "first_generate": 0.9555605999994441,
    "fixed": {
     "1": {
      "warmup": 0.995128400001704,
      "prefill": 0.3198828999993566,
      "total": 6.016274499997962,
      "tokens": 64,
      "forced": true,
      "decode_tps": 11.059632908667202
     },
     "8": {
      "warmup": 0.97472480000215,
      "prefill": 0.34972689999995055,
      "total": 5.494132599997101,
      "tokens": 512,
      "forced": true,
      "decode_tps": 97.97050026600334
     }
    },
    "natural": {
     "seconds": 20.506308299998636,
     "tokens": 1715,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1033592320,
    "detach": 0.040338700000575045
   },
   {
    "name": "0.8b_modellora_adapter",
    "slot": "Lora003",
    "rank": 8,
    "attach": 0.14261330000226735,
    "resting_vram": 853495808,
    "first_generate": 0.9343713000016578,
    "fixed": {
     "1": {
      "warmup": 0.9881807999990997,
      "prefill": 0.3208900000026915,
      "total": 5.972023199999967,
      "tokens": 64,
      "forced": true,
      "decode_tps": 11.148206522548499
     },
     "8": {
      "warmup": 0.9678291999989597,
      "prefill": 0.34688829999868176,
      "total": 5.554986999999528,
      "tokens": 512,
      "forced": true,
      "decode_tps": 96.77235955607333
     }
    },
    "natural": {
     "seconds": 20.551464100000885,
     "tokens": 1697,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1020812800,
    "detach": 0.04006379999918863
   },
   {
    "name": "0.8b_modellora_adapter",
    "slot": "Lora004",
    "rank": 4,
    "attach": 0.12960149999707937,
    "resting_vram": 847106048,
    "first_generate": 0.9424199999994016,
    "fixed": {
     "1": {
      "warmup": 1.0018308000035177,
      "prefill": 0.31913150000036694,
      "total": 6.089435799996863,
      "tokens": 64,
      "forced": true,
      "decode_tps": 10.917968399004234
     },
     "8": {
      "warmup": 0.9696559000003617,
      "prefill": 0.34556229999725474,
      "total": 5.50106550000055,
      "tokens": 512,
      "forced": true,
      "decode_tps": 97.75961345532244
     }
    },
    "natural": {
     "seconds": 20.313275100001192,
     "tokens": 1716,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1014423040,
    "detach": 0.03955330000098911
   },
   {
    "name": "0.8b_modellora_adapter",
    "slot": "Lora005",
    "rank": 32,
    "attach": 0.29407740000169724,
    "resting_vram": 891834368,
    "first_generate": 0.9370025999996869,
    "fixed": {
     "1": {
      "warmup": 0.9960038000026543,
      "prefill": 0.31818530000236933,
      "total": 5.803060400001414,
      "tokens": 64,
      "forced": true,
      "decode_tps": 11.486132109008459
     },
     "8": {
      "warmup": 0.9621183000017481,
      "prefill": 0.34138809999785735,
      "total": 5.475923699999839,
      "tokens": 512,
      "forced": true,
      "decode_tps": 98.15882861924366
     }
    },
    "natural": {
     "seconds": 20.44064100000105,
     "tokens": 1708,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1059151360,
    "detach": 0.04277390000061132
   },
   {
    "resting_vram": 840716288,
    "first_generate": 0.7256110000016633,
    "fixed": {
     "1": {
      "warmup": 0.7887532999993709,
      "prefill": 0.29363089999969816,
      "total": 4.139807899999141,
      "tokens": 64,
      "forced": true,
      "decode_tps": 16.379901392995986
     },
     "8": {
      "warmup": 0.7714802000009513,
      "prefill": 0.31745519999822136,
      "total": 4.041273700000602,
      "tokens": 512,
      "forced": true,
      "decode_tps": 135.3449422950334
     }
    },
    "natural": {
     "seconds": 14.956894599999941,
     "tokens": 1819,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1008033280,
    "name": "base_again",
    "slot": null,
    "rank": null
   }
  ],
  "environment": {
   "python": "3.11.9",
   "torch": "2.11.0+cu128",
   "transformers": "5.5.0",
   "peft": "0.20.0",
   "unsloth": null,
   "gpu": "NVIDIA RTX A6000",
   "gpu_memory": 51526500352
  },
  "cold": {
   "import": 4.950453400000697,
   "model_load": 12.057280299999547,
   "inference_setup": 0.0013660999975400046,
   "vram_after_load": 832196096,
   "first_generate": 1.1290845000003173
  },
  "process_seconds": 413.5576144999977,
  "gpu_before": {
   "utilization": 0.3333333333333333,
   "memory_used": 1331691520.0
  }
 },
 {
  "base_model": "unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit",
  "loader": "unsloth",
  "ok": true,
  "configs": [
   {
    "resting_vram": 568479744,
    "first_generate": 0.8738097999994352,
    "fixed": {
     "1": {
      "warmup": 0.341927500001475,
      "prefill": 0.04913479999959236,
      "total": 2.455972199997632,
      "tokens": 64,
      "forced": true,
      "decode_tps": 26.175428385835836
     },
     "8": {
      "warmup": 0.4819463000021642,
      "prefill": 0.0563469000007899,
      "total": 3.37212460000228,
      "tokens": 512,
      "forced": true,
      "decode_tps": 152.00053972248304
     }
    },
    "natural": {
     "seconds": 13.042687499997555,
     "tokens": 1497,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 677107200,
    "name": "base",
    "slot": null,
    "rank": null
   },
   {
    "name": "QWen2.5-0.5b-lora_adapter",
    "slot": "Lora001",
    "rank": 16,
    "attach": 0.3442557999987912,
    "resting_vram": 673626112,
    "first_generate": 0.4841386000007333,
    "fixed": {
     "1": {
      "warmup": 0.45694020000155433,
      "prefill": 0.09813489999942249,
      "total": 3.277689200000168,
      "tokens": 64,
      "forced": true,
      "decode_tps": 19.814097843834663
     },
     "8": {
      "warmup": 0.5997733999975026,
      "prefill": 0.10366440000143484,
      "total": 4.443024100000912,
      "tokens": 512,
      "forced": true,
      "decode_tps": 116.14616783210222
     }
    },
    "natural": {
     "seconds": 9.197397200001433,
     "tokens": 282,
     "longest": 133,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 723721216,
    "detach": 0.033808500000304775
   },
   {
    "name": "QWen2.5-0.5b-lora_adapter",
    "slot": "Lora002",
    "rank": 16,
    "attach": 0.34501450000243494,
    "resting_vram": 673626112,
    "first_generate": 0.48215820000041276,
    "fixed": {
     "1": {
      "warmup": 0.45601609999721404,
      "prefill": 0.09739909999916563,
      "total": 3.2889656000006653,
      "tokens": 64,
      "forced": true,
      "decode_tps": 19.739522895722335
     },
     "8": {
      "warmup": 0.5780854000004183,
      "prefill": 0.11254600000029313,
      "total": 4.430776900000637,
      "tokens": 512,
      "forced": true,
      "decode_tps": 116.71446286023284
     }
    },
    "natural": {
     "seconds": 9.190880999998626,
     "tokens": 282,
     "longest": 133,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 723721216,
    "detach": 0.03304519999801414
   },
   {
    "name": "QWen2.5-0.5b-lora_adapter",
    "slot": "Lora003",
    "rank": 8,
    "attach": 0.27973240000210353,
    "resting_vram": 656029696,
    "first_generate": 0.49354160000075353,
    "fixed": {
     "1": {
      "warmup": 0.4497239999982412,
      "prefill": 0.099032599999191,
      "total": 3.2711397999992187,
      "tokens": 64,
      "forced": true,
      "decode_tps": 19.860615051092676
     },
     "8": {
      "warmup": 0.6660599000024376,
      "prefill": 0.1055765999990399,
      "total": 4.562833500000124,
      "tokens": 512,
      "forced": true,
      "decode_tps": 113.07402990388043
     }
    },
    "natural": {
     "seconds": 5.2576827999982925,
     "tokens": 295,
     "longest": 75,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 697326592,
    "detach": 0.03410250000160886
   },
   {
    "name": "QWen2.5-0.5b-lora_adapter",
    "slot": "Lora004",
    "rank": 4,
    "attach": 0.19706469999800902,
    "resting_vram": 647231488,
    "first_generate": 0.47726200000033714,
    "fixed": {
     "1": {
      "warmup": 0.46652869999888935,
      "prefill": 0.09748319999926025,
      "total": 3.382533100000728,
      "tokens": 64,
      "forced": true,
      "decode_tps": 19.177790876166554
     },
     "8": {
      "warmup": 0.5956460000015795,
      "prefill": 0.10528720000002068,
      "total": 4.504518099998677,
      "tokens": 512,
      "forced": true,
      "decode_tps": 114.5654800706537
     }
    },
    "natural": {
     "seconds": 5.948787400000583,
     "tokens": 269,
     "longest": 85,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 684129280,
    "detach": 0.043254099997284357
   },
   {
    "name": "QWen2.5-0.5b-lora_adapter",
    "slot": "Lora005",
    "rank": 32,
    "attach": 0.46365870000227005,
    "resting_vram": 708818944,
    "first_generate": 0.4875176999994437,
    "fixed": {
     "1": {
      "warmup": 0.44686349999756203,
      "prefill": 0.0992675000015879,
      "total": 3.2792716000003566,
      "tokens": 64,
      "forced": true,
      "decode_tps": 19.811295211859758
     },
     "8": {
      "warmup": 0.5981410000022152,
      "prefill": 0.10665830000289134,
      "total": 4.436008600001514,
      "tokens": 512,
      "forced": true,
      "decode_tps": 116.41469621900549
     }
    },
    "natural": {
     "seconds": 16.559617000002618,
     "tokens": 488,
     "longest": 239,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 781743616,
    "detach": 0.03377420000106213
   },
   {
    "resting_vram": 638433280,
    "first_generate": 0.3157541000000492,
    "fixed": {
     "1": {
      "warmup": 0.32281370000055176,
      "prefill": 0.04985560000204714,
      "total": 2.432265200000984,
      "tokens": 64,
      "forced": true,
      "decode_tps": 26.443815538700026
     },
     "8": {
      "warmup": 0.4160245999992185,
      "prefill": 0.059263799998007016,
      "total": 3.350485299997672,
      "tokens": 512,
      "forced": true,
      "decode_tps": 153.13463405609477
     }
    },
    "natural": {
     "seconds": 13.02159219999885,
     "tokens": 1497,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 677107200,
    "name": "base_again",
    "slot": null,
    "rank": null
   }
  ],
  "environment": {
   "python": "3.11.9",
   "torch": "2.11.0+cu128",
   "transformers": "5.5.0",
   "peft": "0.20.0",
   "unsloth": "2026.8.15",
   "gpu": "NVIDIA RTX A6000",
   "gpu_memory": 51526500352
  },
  "cold": {
   "import": 9.554969400000118,
   "model_load": 9.157865799999854,
   "inference_setup": 0.002646400000230642,
   "vram_after_load": 568479744,
   "first_generate": 0.8738097999994352
  },
  "process_seconds": 264.37693070000023,
  "gpu_before": {
   "utilization": 0.0,
   "memory_used": 1287651328.0
  }
 },
 {
  "base_model": "unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit",
  "loader": "plain",
  "ok": true,
  "configs": [
   {
    "resting_vram": 537776128,
    "first_generate": 0.7000241999994614,
    "fixed": {
     "1": {
      "warmup": 0.35683740000240505,
      "prefill": 0.05819070000143256,
      "total": 2.729024300002493,
      "tokens": 64,
      "forced": true,
      "decode_tps": 23.588141170597446
     },
     "8": {
      "warmup": 0.36657969999942,
      "prefill": 0.05456940000294708,
      "total": 2.511931099998037,
      "tokens": 512,
      "forced": true,
      "decode_tps": 205.09801223035544
     }
    },
    "natural": {
     "seconds": 9.66767670000263,
     "tokens": 1481,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 594042880,
    "name": "base",
    "slot": null,
    "rank": null
   },
   {
    "name": "QWen2.5-0.5b-lora_adapter",
    "slot": "Lora001",
    "rank": 16,
    "attach": 0.48152249999839114,
    "resting_vram": 582537216,
    "first_generate": 0.7156991000010748,
    "fixed": {
     "1": {
      "warmup": 0.6762952999997651,
      "prefill": 0.09843069999988074,
      "total": 5.3996337999997195,
      "tokens": 64,
      "forced": true,
      "decode_tps": 11.884094763319277
     },
     "8": {
      "warmup": 0.6944869999970251,
      "prefill": 0.09630420000030426,
      "total": 4.994337399999495,
      "tokens": 512,
      "forced": true,
      "decode_tps": 102.89844503301515
     }
    },
    "natural": {
     "seconds": 11.183062100000825,
     "tokens": 329,
     "longest": 144,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 621218816,
    "detach": 0.038330399998812936
   },
   {
    "name": "QWen2.5-0.5b-lora_adapter",
    "slot": "Lora002",
    "rank": 16,
    "attach": 0.3405429999984335,
    "resting_vram": 582537216,
    "first_generate": 0.6989905000009458,
    "fixed": {
     "1": {
      "warmup": 0.6767666999985522,
      "prefill": 0.09599110000272049,
      "total": 5.395142600002146,
      "tokens": 64,
      "forced": true,
      "decode_tps": 11.888695765729068
     },
     "8": {
      "warmup": 0.6886013000003004,
      "prefill": 0.09585240000160411,
      "total": 4.994274799999403,
      "tokens": 512,
      "forced": true,
      "decode_tps": 102.89026932430868
     }
    },
    "natural": {
     "seconds": 11.1906670999997,
     "tokens": 329,
     "longest": 144,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 621218816,
    "detach": 0.035865300000295974
   },
   {
    "name": "QWen2.5-0.5b-lora_adapter",
    "slot": "Lora003",
    "rank": 8,
    "attach": 0.2897448000003351,
    "resting_vram": 564940800,
    "first_generate": 0.7097634999990987,
    "fixed": {
     "1": {
      "warmup": 0.6801183999996283,
      "prefill": 0.09567539999989094,
      "total": 5.498883099997329,
      "tokens": 64,
      "forced": true,
      "decode_tps": 11.659740564855554
     },
     "8": {
      "warmup": 0.6445013999982621,
      "prefill": 0.09673550000297837,
      "total": 5.059192599997914,
      "tokens": 512,
      "forced": true,
      "decode_tps": 101.56259083842042
     }
    },
    "natural": {
     "seconds": 19.67901620000339,
     "tokens": 495,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 611639296,
    "detach": 0.03832239999974263
   },
   {
    "name": "QWen2.5-0.5b-lora_adapter",
    "slot": "Lora004",
    "rank": 4,
    "attach": 0.19815309999830788,
    "resting_vram": 556142592,
    "first_generate": 0.6765434000008099,
    "fixed": {
     "1": {
      "warmup": 0.6913793999992777,
      "prefill": 0.09712939999735681,
      "total": 5.43613170000026,
      "tokens": 64,
      "forced": true,
      "decode_tps": 11.79995745646443
     },
     "8": {
      "warmup": 0.6941574000011315,
      "prefill": 0.09446810000008554,
      "total": 4.995802500001446,
      "tokens": 512,
      "forced": true,
      "decode_tps": 102.8291397542392
     }
    },
    "natural": {
     "seconds": 6.44085129999803,
     "tokens": 267,
     "longest": 83,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 594824192,
    "detach": 0.03376340000249911
   },
   {
    "name": "QWen2.5-0.5b-lora_adapter",
    "slot": "Lora005",
    "rank": 32,
    "attach": 0.42644020000079763,
    "resting_vram": 617730048,
    "first_generate": 0.709743499999604,
    "fixed": {
     "1": {
      "warmup": 0.686539400001493,
      "prefill": 0.09764890000224113,
      "total": 5.407704900000681,
      "tokens": 64,
      "forced": true,
      "decode_tps": 11.864281657296742
     },
     "8": {
      "warmup": 0.68333469999925,
      "prefill": 0.09685909999825526,
      "total": 4.996171500002674,
      "tokens": 512,
      "forced": true,
      "decode_tps": 102.8715784687552
     }
    },
    "natural": {
     "seconds": 18.66089009999996,
     "tokens": 492,
     "longest": 240,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 663155200,
    "detach": 0.03610590000243974
   },
   {
    "resting_vram": 547344384,
    "first_generate": 0.3592399999979534,
    "fixed": {
     "1": {
      "warmup": 0.34284780000234605,
      "prefill": 0.057245899999543326,
      "total": 2.7627695000010135,
      "tokens": 64,
      "forced": true,
      "decode_tps": 23.285695973957043
     },
     "8": {
      "warmup": 0.33283859999937704,
      "prefill": 0.05787839999902644,
      "total": 2.5145942000017385,
      "tokens": 512,
      "forced": true,
      "decode_tps": 205.15193495293335
     }
    },
    "natural": {
     "seconds": 9.734685299998091,
     "tokens": 1481,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 594042880,
    "name": "base_again",
    "slot": null,
    "rank": null
   }
  ],
  "environment": {
   "python": "3.11.9",
   "torch": "2.11.0+cu128",
   "transformers": "5.5.0",
   "peft": "0.20.0",
   "unsloth": null,
   "gpu": "NVIDIA RTX A6000",
   "gpu_memory": 51526500352
  },
  "cold": {
   "import": 4.908867399997689,
   "model_load": 4.790170399999624,
   "inference_setup": 0.0009046999985002913,
   "vram_after_load": 537776128,
   "first_generate": 0.7000241999994614
  },
  "process_seconds": 308.2773488999992,
  "gpu_before": {
   "utilization": 0.6666666666666666,
   "memory_used": 1278214144.0
  }
 },
 {
  "base_model": "unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit",
  "loader": "unsloth",
  "ok": true,
  "configs": [
   {
    "resting_vram": 1578124288,
    "first_generate": 1.0734157000006235,
    "fixed": {
     "1": {
      "warmup": 0.37601310000172816,
      "prefill": 0.0605699999978242,
      "total": 2.7812449999983073,
      "tokens": 64,
      "forced": true,
      "decode_tps": 23.15601826752141
     },
     "8": {
      "warmup": 0.5133499999974447,
      "prefill": 0.0672078000025067,
      "total": 3.833267400001205,
      "tokens": 512,
      "forced": true,
      "decode_tps": 133.82687836384062
     }
    },
    "natural": {
     "seconds": 14.904038800003036,
     "tokens": 1576,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1799330304,
    "name": "base",
    "slot": null,
    "rank": null
   },
   {
    "name": "Qwen2.5-1.5B-Instruct-medical-lora_adapter",
    "slot": "Lora001",
    "rank": 4,
    "attach": 0.33075169999938225,
    "resting_vram": 1758372864,
    "first_generate": 0.5776218999999401,
    "fixed": {
     "1": {
      "warmup": 0.53730060000089,
      "prefill": 0.11498690000007628,
      "total": 3.8345614000027126,
      "tokens": 64,
      "forced": true,
      "decode_tps": 16.937421202332512
     },
     "8": {
      "warmup": 0.7083103000004485,
      "prefill": 0.12024139999994077,
      "total": 5.263938599997346,
      "tokens": 512,
      "forced": true,
      "decode_tps": 97.98399485884478
     }
    },
    "natural": {
     "seconds": 1.5995774999973946,
     "tokens": 109,
     "longest": 19,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1808809984,
    "detach": 0.04155760000139708
   },
   {
    "name": "Qwen2.5-1.5B-Instruct-medical-lora_adapter",
    "slot": "Lora002",
    "rank": 8,
    "attach": 0.38125510000099894,
    "resting_vram": 1776837632,
    "first_generate": 0.56309390000024,
    "fixed": {
     "1": {
      "warmup": 0.5504313000019465,
      "prefill": 0.11712249999982305,
      "total": 3.8397369000012986,
      "tokens": 64,
      "forced": true,
      "decode_tps": 16.92359004466727
     },
     "8": {
      "warmup": 0.6893327000034333,
      "prefill": 0.12325050000072224,
      "total": 5.266419100000348,
      "tokens": 512,
      "forced": true,
      "decode_tps": 97.99406537052599
     }
    },
    "natural": {
     "seconds": 5.043069600000308,
     "tokens": 172,
     "longest": 62,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1835032576,
    "detach": 0.04332000000067637
   },
   {
    "name": "Qwen2.5-1.5B-Instruct-medical-lora_adapter",
    "slot": "Lora003",
    "rank": 16,
    "attach": 0.4779660000021977,
    "resting_vram": 1813767168,
    "first_generate": 0.6228855999979714,
    "fixed": {
     "1": {
      "warmup": 0.5917421000012837,
      "prefill": 0.12022280000019236,
      "total": 3.918695000000298,
      "tokens": 64,
      "forced": true,
      "decode_tps": 16.585615658842585
     },
     "8": {
      "warmup": 0.7030733000028704,
      "prefill": 0.1360741999997117,
      "total": 5.277860700000019,
      "tokens": 512,
      "forced": true,
      "decode_tps": 98.02040594255904
     }
    },
    "natural": {
     "seconds": 3.0069414999998116,
     "tokens": 123,
     "longest": 35,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1890426880,
    "detach": 0.04287839999960852
   },
   {
    "name": "Qwen2.5-1.5B-Instruct-medical-lora_adapter",
    "slot": "Lora004",
    "rank": 32,
    "attach": 0.6766381999987061,
    "resting_vram": 1893721088,
    "first_generate": 0.5975954000023194,
    "fixed": {
     "1": {
      "warmup": 0.5376500999991549,
      "prefill": 0.11986149999938789,
      "total": 3.950312400000257,
      "tokens": 64,
      "forced": true,
      "decode_tps": 16.447149864258982
     },
     "8": {
      "warmup": 0.8239588000033109,
      "prefill": 0.12418400000024121,
      "total": 5.271054199998616,
      "tokens": 512,
      "forced": true,
      "decode_tps": 97.92358859179295
     }
    },
    "natural": {
     "seconds": 2.852607299999363,
     "tokens": 122,
     "longest": 35,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 2007310336,
    "detach": 0.04088430000047083
   },
   {
    "name": "Qwen2.5-1.5B-Instruct-medical-lora_adapter",
    "slot": "Lora005",
    "rank": 8,
    "attach": 0.3847600000008242,
    "resting_vram": 1776837632,
    "first_generate": 0.5724401000006765,
    "fixed": {
     "1": {
      "warmup": 0.5395016000002215,
      "prefill": 0.1155481999994663,
      "total": 3.8757865999978094,
      "tokens": 64,
      "forced": true,
      "decode_tps": 16.754256857764062
     },
     "8": {
      "warmup": 0.7089459999988321,
      "prefill": 0.12049780000234023,
      "total": 5.19528949999949,
      "tokens": 512,
      "forced": true,
      "decode_tps": 99.3144211220104
     }
    },
    "natural": {
     "seconds": 1.4244451000013214,
     "tokens": 95,
     "longest": 17,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1835032576,
    "detach": 0.03900910000083968
   },
   {
    "resting_vram": 1739908096,
    "first_generate": 0.3709183999999368,
    "fixed": {
     "1": {
      "warmup": 0.3791592000015953,
      "prefill": 0.05700290000095265,
      "total": 2.7637317000007897,
      "tokens": 64,
      "forced": true,
      "decode_tps": 23.27532776833933
     },
     "8": {
      "warmup": 0.4898858000015025,
      "prefill": 0.06382830000075046,
      "total": 3.8214152000000468,
      "tokens": 512,
      "forced": true,
      "decode_tps": 134.12863452342097
     }
    },
    "natural": {
     "seconds": 14.935846600001241,
     "tokens": 1576,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1799330304,
    "name": "base_again",
    "slot": null,
    "rank": null
   }
  ],
  "environment": {
   "python": "3.11.9",
   "torch": "2.11.0+cu128",
   "transformers": "5.5.0",
   "peft": "0.20.0",
   "unsloth": "2026.8.15",
   "gpu": "NVIDIA RTX A6000",
   "gpu_memory": 51526500352
  },
  "cold": {
   "import": 9.540245199998026,
   "model_load": 9.820315500001016,
   "inference_setup": 0.0030490999997709878,
   "vram_after_load": 1578124288,
   "first_generate": 1.0734157000006235
  },
  "process_seconds": 266.46203930000047,
  "gpu_before": {
   "utilization": 0.6666666666666666,
   "memory_used": 1218445312.0
  }
 },
 {
  "base_model": "unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit",
  "loader": "plain",
  "ok": true,
  "configs": [
   {
    "resting_vram": 1527333888,
    "first_generate": 0.7401533000011113,
    "fixed": {
     "1": {
      "warmup": 0.4119165999982215,
      "prefill": 0.06662359999972978,
      "total": 3.092130199998792,
      "tokens": 64,
      "forced": true,
      "decode_tps": 20.822959037676377
     },
     "8": {
      "warmup": 0.3783231000015803,
      "prefill": 0.06171360000007553,
      "total": 2.9113232999989123,
      "tokens": 512,
      "forced": true,
      "decode_tps": 176.8663266412259
     }
    },
    "natural": {
     "seconds": 11.332786099999794,
     "tokens": 1372,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1636706816,
    "name": "base",
    "slot": null,
    "rank": null
   },
   {
    "name": "Qwen2.5-1.5B-Instruct-medical-lora_adapter",
    "slot": "Lora001",
    "rank": 4,
    "attach": 0.4989124999992782,
    "resting_vram": 1555366912,
    "first_generate": 0.8017827000003308,
    "fixed": {
     "1": {
      "warmup": 0.7805488999983936,
      "prefill": 0.11111589999927673,
      "total": 6.084509199998138,
      "tokens": 64,
      "forced": true,
      "decode_tps": 10.546769120327639
     },
     "8": {
      "warmup": 0.8424015000018699,
      "prefill": 0.11747599999944214,
      "total": 5.761552100000699,
      "tokens": 512,
      "forced": true,
      "decode_tps": 89.29716592586124
     }
    },
    "natural": {
     "seconds": 1.727758699998958,
     "tokens": 109,
     "longest": 19,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1609998848,
    "detach": 0.04044909999720403
   },
   {
    "name": "Qwen2.5-1.5B-Instruct-medical-lora_adapter",
    "slot": "Lora002",
    "rank": 8,
    "attach": 0.3835658999996667,
    "resting_vram": 1573831680,
    "first_generate": 0.8025386999979673,
    "fixed": {
     "1": {
      "warmup": 0.7685664999989967,
      "prefill": 0.11591050000060932,
      "total": 6.04540310000084,
      "tokens": 64,
      "forced": true,
      "decode_tps": 10.624855152023894
     },
     "8": {
      "warmup": 0.741035600000032,
      "prefill": 0.11137299999973038,
      "total": 5.743991200000892,
      "tokens": 512,
      "forced": true,
      "decode_tps": 89.47881466560189
     }
    },
    "natural": {
     "seconds": 5.6126177000005555,
     "tokens": 187,
     "longest": 62,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1628463616,
    "detach": 0.04368040000190376
   },
   {
    "name": "Qwen2.5-1.5B-Instruct-medical-lora_adapter",
    "slot": "Lora003",
    "rank": 16,
    "attach": 0.495046200001525,
    "resting_vram": 1610761216,
    "first_generate": 0.8003501999992295,
    "fixed": {
     "1": {
      "warmup": 0.7790801000010106,
      "prefill": 0.11030910000044969,
      "total": 6.07386150000093,
      "tokens": 64,
      "forced": true,
      "decode_tps": 10.564173126070783
     },
     "8": {
      "warmup": 0.7464906999994128,
      "prefill": 0.11082499999974971,
      "total": 5.76295939999909,
      "tokens": 512,
      "forced": true,
      "decode_tps": 89.16985413511377
     }
    },
    "natural": {
     "seconds": 3.6047155000014754,
     "tokens": 147,
     "longest": 40,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1665393152,
    "detach": 0.04083539999919594
   },
   {
    "name": "Qwen2.5-1.5B-Instruct-medical-lora_adapter",
    "slot": "Lora004",
    "rank": 32,
    "attach": 0.7391922000024351,
    "resting_vram": 1685996544,
    "first_generate": 0.8043623000012303,
    "fixed": {
     "1": {
      "warmup": 0.788944600000832,
      "prefill": 0.11382030000095256,
      "total": 6.093365400000039,
      "tokens": 64,
      "forced": true,
      "decode_tps": 10.535918526646721
     },
     "8": {
      "warmup": 0.7606369000022823,
      "prefill": 0.11483520000183489,
      "total": 5.889966499998991,
      "tokens": 512,
      "forced": true,
      "decode_tps": 87.27074309120006
     }
    },
    "natural": {
     "seconds": 3.2336928000004264,
     "tokens": 122,
     "longest": 35,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1740628480,
    "detach": 0.04528600000048755
   },
   {
    "name": "Qwen2.5-1.5B-Instruct-medical-lora_adapter",
    "slot": "Lora005",
    "rank": 8,
    "attach": 0.3864681000013661,
    "resting_vram": 1573831680,
    "first_generate": 0.8238228000009258,
    "fixed": {
     "1": {
      "warmup": 0.7922493000005488,
      "prefill": 0.11186349999843515,
      "total": 6.056058199999825,
      "tokens": 64,
      "forced": true,
      "decode_tps": 10.598576119989016
     },
     "8": {
      "warmup": 0.7569583000004059,
      "prefill": 0.11270780000268132,
      "total": 5.800785700001143,
      "tokens": 512,
      "forced": true,
      "decode_tps": 88.60638142809829
     }
    },
    "natural": {
     "seconds": 1.5545383999997284,
     "tokens": 95,
     "longest": 17,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1628463616,
    "detach": 0.040757500002655433
   },
   {
    "resting_vram": 1536902144,
    "first_generate": 0.41336479999881703,
    "fixed": {
     "1": {
      "warmup": 0.4039364000018395,
      "prefill": 0.06538819999695988,
      "total": 3.0827759000021615,
      "tokens": 64,
      "forced": true,
      "decode_tps": 20.878987476449048
     },
     "8": {
      "warmup": 0.38661310000316007,
      "prefill": 0.06853819999741972,
      "total": 2.9341993000016373,
      "tokens": 512,
      "forced": true,
      "decode_tps": 175.87564698395713
     }
    },
    "natural": {
     "seconds": 11.340350900001795,
     "tokens": 1372,
     "longest": 250,
     "calls": 1,
     "prompts": 8
    },
    "peak_vram": 1636706816,
    "name": "base_again",
    "slot": null,
    "rank": null
   }
  ],
  "environment": {
   "python": "3.11.9",
   "torch": "2.11.0+cu128",
   "transformers": "5.5.0",
   "peft": "0.20.0",
   "unsloth": null,
   "gpu": "NVIDIA RTX A6000",
   "gpu_memory": 51526500352
  },
  "cold": {
   "import": 4.872376199997234,
   "model_load": 5.107083599999896,
   "inference_setup": 0.0008825000004435424,
   "vram_after_load": 1527333888,
   "first_generate": 0.7401533000011113
  },
  "process_seconds": 290.3799003999993,
  "gpu_before": {
   "utilization": 0.3333333333333333,
   "memory_used": 1219493888.0
  }
 }
]
```

</details>
