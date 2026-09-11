# GLM-5.3 Flash EXL3 — single GB10 profile

GLM-5.3 Flash running locally on one NVIDIA GB10 machine.

| Setting | Value |
|---|---|
| Quantization | EXL3 2.05 bpw |
| Weight size | approximately 85.2 GB |
| Maximum context | 262,144 tokens |
| KV cache | FP8, approximately 371,700 tokens |
| Concurrent sequences | up to 3 |
| Acceleration | DFlash2 K5 and prefix caching |
| Features | reasoning, coding, tools, JSON, images |
| Served model ID | `GLM-5.3-Flash-EXL3` |

Reference single-stream throughput for the same K5 recipe:

| Workload | Approximate throughput |
|---|---:|
| Open-ended prose | 29.9 tok/s |
| Code | 40.1 tok/s |
| Structured output | 53.5 tok/s |
| Large-context prefill | approximately 800 tok/s |

During live prefix-cache qualification, a repeated 35,840-token prefix reduced
request time from approximately 52 seconds to 1.6 seconds with no EngineCore
restart. Actual speed varies with prompt length, sampling, thinking mode, cache
reuse, and concurrent load.

This profile uses the standard checkpoint, not an abliterated or uncensored
variant. It supports up to two images per request; video is disabled.
