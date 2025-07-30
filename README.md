# Results from A100-4GB PCI-e

## Profiling Single Precision

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 51.0 | 919,138 | 2 | 459,569.0 | 459,569.0 | 458,721 | 460,417 | 1,199.3 | `stencil(int, int, float *, float *)` |
| 49.0 | 881,603 | 2 | 440,801.5 | 440,801.5 | 440,737 | 440,866 | 91.2 | `copy(int, int, float *, float *)` |

## Profiling Half Precision

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 51.7 | 807,907 | 2 | 403,953.5 | 403,953.5 | 402,498 | 405,409 | 2,058.4 | `stencil(int, int, __half *, __half *)` |
| 48.3 | 755,554 | 2 | 377,777.0 | 377,777.0 | 377,665 | 377,889 | 158.4 | `copy(int, int, __half *, __half *)` |

## Profiling Vectorized Float

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 50.2 | 768,098 | 2 | 384,049.0 | 384,049.0 | 383,937 | 384,161 | 158.4 | `copy_vector(int, int, float4 *, float4 *)` |
| 49.8 | 761,378 | 2 | 380,689.0 | 380,689.0 | 374,369 | 387,009 | 8,937.8 | `stencil_vector(int, int, float4 *, float4 *)` |

## Profiling Vectorized Half Precision

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 51.9 | 489,858 | 2 | 244,929.0 | 244,929.0 | 243,617 | 246,241 | 1,855.4 | `stencil_vector_half2(int, int, __half2 *, __half2 *)` |
| 48.1 | 453,921 | 2 | 226,960.5 | 226,960.5 | 226,817 | 227,104 | 202.9 | `copy_vector_half(int, int, __half2 *, __half2 *)` |

# Results from H100-80GB PCI-e

## Profiling Single Precision

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 51.5 | 730,788 | 2 | 365,394.0 | 365,394.0 | 362,626 | 368,162 | 3,914.5 | `stencil(int, int, float *, float *)` |
| 48.5 | 689,253 | 2 | 344,626.5 | 344,626.5 | 344,386 | 344,867 | 340.1 | `copy(int, int, float *, float *)` |

## Profiling Half Precision

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 51.7 | 608,100 | 2 | 304,050.0 | 304,050.0 | 302,434 | 305,666 | 2,285.4 | `stencil(int, int, __half *, __half *)` |
| 48.3 | 568,900 | 2 | 284,450.0 | 284,450.0 | 284,194 | 284,706 | 362.0 | `copy(int, int, __half *, __half *)` |

## Profiling Vectorized Float

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 50.8 | 589,124 | 2 | 294,562.0 | 294,562.0 | 294,338 | 294,786 | 316.8 | `copy_vector(int, int, float4 *, float4 *)` |
| 49.2 | 571,363 | 2 | 285,681.5 | 285,681.5 | 277,025 | 294,338 | 12,242.1 | `stencil_vector(int, int, float4 *, float4 *)` |

## Profiling Vectorized Half Precision

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 51.5 | 364,418 | 2 | 182,209.0 | 182,209.0 | 180,993 | 183,425 | 1,719.7 | `stencil_vector_half2(int, int, __half2 *, __half2 *)` |
| 48.5 | 342,850 | 2 | 171,425.0 | 171,425.0 | 171,329 | 171,521 | 135.8 | `copy_vector_half(int, int, __half2 *, __half2 *)` |
