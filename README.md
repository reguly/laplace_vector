# Results from A100-4GB PCI-e
Profiling Single Precision...
----------------------------------------
Single Precision Kernel Statistics:
----------------------------------------

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                 Name                
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -----------------------------------
     51.0          919,138          2  459,569.0  459,569.0   458,721   460,417      1,199.3  stencil(int, int, float *, float *)
     49.0          881,603          2  440,801.5  440,801.5   440,737   440,866         91.2  copy(int, int, float *, float *)   


Profiling Half Precision...
----------------------------------------
Half Precision Kernel Statistics:
----------------------------------------

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                  Name                 
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------------------------------------
     51.7          807,907          2  403,953.5  403,953.5   402,498   405,409      2,058.4  stencil(int, int, __half *, __half *)
     48.3          755,554          2  377,777.0  377,777.0   377,665   377,889        158.4  copy(int, int, __half *, __half *)   


Profiling Vectorized Float...
----------------------------------------
Vectorized Float Kernel Statistics:
----------------------------------------

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                      Name                    
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  --------------------------------------------
     50.2          768,098          2  384,049.0  384,049.0   383,937   384,161        158.4  copy_vector(int, int, float4 *, float4 *)   
     49.8          761,378          2  380,689.0  380,689.0   374,369   387,009      8,937.8  stencil_vector(int, int, float4 *, float4 *)


Profiling Vectorized Half Precision...
----------------------------------------
Vectorized Half Precision Kernel Statistics:
----------------------------------------

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                          Name                        
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------
     51.9          489,858          2  244,929.0  244,929.0   243,617   246,241      1,855.4  stencil_vector_half2(int, int, __half2 *, __half2 *)
     48.1          453,921          2  226,960.5  226,960.5   226,817   227,104        202.9  copy_vector_half(int, int, __half2 *, __half2 *)    

# Results from H100-80GB PCI-e

Profiling Single Precision...
----------------------------------------
Single Precision Kernel Statistics:
----------------------------------------

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                 Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -----------------------------------
     51.5           730788          2  365394.0  365394.0    362626    368162       3914.5  stencil(int, int, float *, float *)
     48.5           689253          2  344626.5  344626.5    344386    344867        340.1  copy(int, int, float *, float *)


Profiling Half Precision...
----------------------------------------
Half Precision Kernel Statistics:
----------------------------------------

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                  Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -------------------------------------
     51.7           608100          2  304050.0  304050.0    302434    305666       2285.4  stencil(int, int, __half *, __half *)
     48.3           568900          2  284450.0  284450.0    284194    284706        362.0  copy(int, int, __half *, __half *)


Profiling Vectorized Float...
----------------------------------------
Vectorized Float Kernel Statistics:
----------------------------------------

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                      Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  --------------------------------------------
     50.8           589124          2  294562.0  294562.0    294338    294786        316.8  copy_vector(int, int, float4 *, float4 *)
     49.2           571363          2  285681.5  285681.5    277025    294338      12242.1  stencil_vector(int, int, float4 *, float4 *)


Profiling Vectorized Half Precision...
----------------------------------------
Vectorized Half Precision Kernel Statistics:
----------------------------------------

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                          Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------
     51.5           364418          2  182209.0  182209.0    180993    183425       1719.7  stencil_vector_half2(int, int, __half2 *, __half2 *)
     48.5           342850          2  171425.0  171425.0    171329    171521        135.8  copy_vector_half(int, int, __half2 *, __half2 *)


# H100-80GB PCI-e

Profiling Single Precision...
----------------------------------------
Single Precision Kernel Statistics:
----------------------------------------

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                 Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -----------------------------------
     51.5           730788          2  365394.0  365394.0    362626    368162       3914.5  stencil(int, int, float *, float *)
     48.5           689253          2  344626.5  344626.5    344386    344867        340.1  copy(int, int, float *, float *)


Profiling Half Precision...
----------------------------------------
Half Precision Kernel Statistics:
----------------------------------------

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                  Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -------------------------------------
     51.7           608100          2  304050.0  304050.0    302434    305666       2285.4  stencil(int, int, __half *, __half *)
     48.3           568900          2  284450.0  284450.0    284194    284706        362.0  copy(int, int, __half *, __half *)


Profiling Vectorized Float...
----------------------------------------
Vectorized Float Kernel Statistics:
----------------------------------------

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                      Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  --------------------------------------------
     50.8           589124          2  294562.0  294562.0    294338    294786        316.8  copy_vector(int, int, float4 *, float4 *)
     49.2           571363          2  285681.5  285681.5    277025    294338      12242.1  stencil_vector(int, int, float4 *, float4 *)


Profiling Vectorized Half Precision...
----------------------------------------
Vectorized Half Precision Kernel Statistics:
----------------------------------------

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                          Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------
     51.5           364418          2  182209.0  182209.0    180993    183425       1719.7  stencil_vector_half2(int, int, __half2 *, __half2 *)
     48.5           342850          2  171425.0  171425.0    171329    171521        135.8  copy_vector_half(int, int, __half2 *, __half2 *)


