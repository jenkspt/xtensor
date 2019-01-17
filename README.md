SpaceX Style Tensor Compression (XTensor)
========================================

This Repo is an example (or my interpretation) of the adaptive grid compression shown
in the following SpaceX video:

[[Youtube Video]](https://youtu.be/txk-VO1hzBY?t=941)  
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/txk-VO1hzBY/0.jpg)](https://youtu.be/txk-VO1hzBY?t=941)  

[xt.py](xt.py) runs the compression on a grayscale image [cat.jpg](cat.jpg) 
using nearest neighbor  (the video shows linear interpolation).

The trick to make this fast is using numpy's vectorized operations.
Anchor points can be selected by striding the input array in both dimensions. 
Then we can simply resize this anchor-point array back up to its original
size using the desired interpolation

```python
anchor = arr[::s, ::s]
n0, n1 = anchor.shape

interp = zoom(anchor[::2,::2], 2, order=order)[:n0,:n1]
```

Next we save the error values that exceed epsilon.

```python
error = anchor - interp

mask = np.abs(error) > epsilon
# Save error values that exceed epsilon
compressed[::s,::s][mask] = error[mask]
```

And then repeat with stride `s *= 2`

The result is a small set of anchor points, and the "compressed" image.
This result image resembles a gradient image in that the values shown
represent the areas of greatest change. In order to actually reduce the
memory footprint of the image, we can represent the image, which should contain
mostly zeros, as a sparse matrix. It appears that the SpaceX adaptive grid
uses a sparse representation similar to 'Dictionary of Keys' (DOK). Scipy has a 
dok_matrix type, but unfortunately it does not extend to N dimensional tensors/arrays.


In short, the decompression works by interpolating between the anchor points,
and adding back the stored errors.

## Examples

### Epsilon: 0
|Interpolation|Non-Zero|Compressed|Decompressed|
|:---:        |:---:|:---:     |:---:       |
|nearest      |65.6%|![c-0-0]  |![d-0-0]    |
|bilinear     |74.3%|![c-0-1]  |![d-0-1]    |
|bicubic      |98.4%|![c-0-3]  |![d-0-3]    |

### Epsilon: 2.5
|Interpolation|Non-Zero|Compressed|Decompressed|Error|%|
|:---:        |:---:|:---:     |:---:       |:---:|:---:|
|nearest      |40.3%|![c-2.5-0]  |![d-2.5-0]    |![e-2.5-0]|0.9%|
|bilinear     |35.0%|![c-2.5-1]  |![d-2.5-1]    |![e-2.5-1]|0.7%|
|bicubic      |36.0%|![c-2.5-3]  |![d-2.5-3]    |![e-2.5-3]|0.9%|

### Epsilon: 10
|Interpolation|Non-Zero|Compressed|Decompressed|Error|%|
|:---:        |:---:|:---:     |:---:       |:---:|:---:|
|nearest      |18.7%|![c-10-0]  |![d-10-0]    |![e-10-0]|4.4%|
|bilinear     |13.2%|![c-10-1]  |![d-10-1]    |![e-10-1]|3.9%|
|bicubic      |14.1%|![c-10-3]  |![d-10-3]    |![e-10-3]|4.3%|


### Epsilon: 25
|Interpolation|Non-Zero|Compressed|Decompressed|Error|%|
|:---:        |:---:|:---:     |:---:       |:---:|:---:|
|nearest      |6.7%|![c-25-0]  |![d-25-0]    |![e-25-0]|11.1%|
|bilinear     |3.6%|![c-25-1]  |![d-25-1]    |![e-25-1]|9.1%|
|bicubic      |3.9%|![c-25-3]  |![d-25-3]    |![e-25-3]|10.1%|


[c-0-0]: images/c_eps-0_ord-0.png
[d-0-0]: images/dc_eps-0_ord-0.png 
[e-0-0]: images/error_eps-0_ord-0.png 

[c-0-1]: images/c_eps-0_ord-1.png 
[d-0-1]: images/dc_eps-0_ord-1.png 
[e-0-1]: images/error_eps-0_ord-1.png 

[c-0-3]: images/c_eps-0_ord-3.png 
[d-0-3]: images/dc_eps-0_ord-3.png 
[e-0-3]: images/error_eps-0_ord-3.png 


[c-2.5-0]: images/c_eps-2.5_ord-0.png
[d-2.5-0]: images/dc_eps-2.5_ord-0.png 
[e-2.5-0]: images/error_eps-2.5_ord-0.png 

[c-2.5-1]: images/c_eps-2.5_ord-1.png 
[d-2.5-1]: images/dc_eps-2.5_ord-1.png 
[e-2.5-1]: images/error_eps-2.5_ord-1.png 

[c-2.5-3]: images/c_eps-2.5_ord-3.png 
[d-2.5-3]: images/dc_eps-2.5_ord-3.png 
[e-2.5-3]: images/error_eps-2.5_ord-3.png 


[c-10-0]: images/c_eps-10_ord-0.png 
[d-10-0]: images/dc_eps-10_ord-0.png 
[e-10-0]: images/error_eps-10_ord-0.png 

[c-10-1]: images/c_eps-10_ord-1.png 
[d-10-1]: images/dc_eps-10_ord-1.png 
[e-10-1]: images/error_eps-10_ord-1.png 

[c-10-3]: images/c_eps-10_ord-3.png 
[d-10-3]: images/dc_eps-10_ord-3.png 
[e-10-3]: images/error_eps-10_ord-3.png 


[c-25-0]: images/c_eps-25_ord-0.png
[d-25-0]: images/dc_eps-25_ord-0.png
[e-25-0]: images/error_eps-25_ord-0.png

[c-25-1]: images/c_eps-25_ord-1.png
[d-25-1]: images/dc_eps-25_ord-1.png
[e-25-1]: images/error_eps-25_ord-1.png

[c-25-3]: images/c_eps-25_ord-3.png
[d-25-3]: images/dc_eps-25_ord-3.png
[e-25-3]: images/error_eps-25_ord-3.png

