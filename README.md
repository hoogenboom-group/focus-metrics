# focus-metrics
 Functions to measure the focus level of fluorescence images.


## `fmeasure`
The original Matlab implementation of the focus metrics accompanying [Pertuz et al. (2013)](https://doi.org/10.1016/j.patcog.2012.11.011) from which `fmetrics` was translated.[^1]

**Focus Measure**[^2]
> This function measures the relative degree of focus of an image. Several up-to-date focus measuring algorithms have been implemented and the function supports uint8 or double images. For futher details on each focus measuring algorithm the reader is referred to [Pertuz et al. (2013)](https://doi.org/10.1016/j.patcog.2012.11.011) and the references therein. For further information and datasets, see https://sites.google.com/view/cvia/focus-measure

[^1]: Pertuz, Said, Domenec Puig, and Miguel Angel Garcia. "Analysis of focus measure operators for shape-from-focus." Pattern Recognition 46.5 (2013): 1415-1432.  
[^2]: https://www.mathworks.com/matlabcentral/fileexchange/27314-focus-measure


## `fmetrics`
Implemented focus metrics

|      | Focus Metric               | Reference
| ---- | -------------------------- | ---------
| ACMO | Absolute central moment    | 
| BREN | Brenner's focus measure    | 
| CURV | Image curvature            | 
| GDER | Gaussian derivative        | 
| GLVA | Gray-level variance        | 
| GLVV | Gray-level local variance  | 
| GRAE | Energy of gradient         | 
| GRAT | Thresholded gradient       | 
| GRAS | Squared gradient           | 
| HELM | Helmli's measure           | 
| HISE | Histogram entropy          | 
| LAPE | Energy of Laplacian        | 
| LAPM | Modified Laplacian         | 
| LAPV | Variance of Laplacian      | 
| LAPD | Diagonal Laplacian         | 
| SFIL | Steerable filters-based    | 
| SFRQ | Spatial frequency          | 
| TENG | Tenegrad                   | 
| TENV | Tenengrad variance         | 
| VOLA | Vollat's correlation-based | 
