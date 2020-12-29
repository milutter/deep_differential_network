# Deep Differential Network:
This package provides an implementation of a Deep Differential Network. This 
network architecture is a variant of a fully connected network that in addition 
to computing the function value $`f(\mathbf{x}; \theta)`$ outputs the network Jacobian 
w.r.t. the network input $`\mathbf{x}`$, i.e.,
$`\mathbf{J} = \partial f(\mathbf{x}; \theta) / \partial \mathbf{x}`$. The 
Jacobian can be computed in closed form with machine precision, with minimal 
computational overhead and for any non-linearity using the chain rule. Within the 
implementation we extend the computational graph of a fully connected layer
with an additional path to compute the partial derivative w.r.t. the previous layer.
Chaining these partial derivatives computes the Jacobian. Therefore, the
Jacobian is computed using forward differentiation and the network
parameters $`\theta`$  can be learned by backpropagating through the Jacobian 
using the standard autograd approach. 

Let $`f(\mathbf{x}; \theta)`$ be the deep network with the parameters $`\theta`$ 
and the input $`\mathbf{x}`$, then the Jacobian can be computed using the chain
rule via 
```math
\frac{\partial f(\mathbf{x}; \theta)}{\partial \mathbf{x}} = 
\frac{\partial f(\mathbf{x}; \theta)}{\partial 
\mathbf{h}_{N-1}} \frac{\partial \mathbf{h}_{N-1}}{\partial \mathbf{h}_{N-2}} 
\hspace{5pt} \cdots \hspace{5pt} 
\frac{\partial \mathbf{h}_1}{\partial \mathbf{x}}. 
```
The partial derivative w.r.t. the previous layer can be computed for 
the fully connected layer, i.e., $`\mathbf{h}_i = g(\mathbf{W}^{T}_{i} 
\mathbf{h}_{i-1} + \mathbf{b})`$, via
```math
\hspace{50pt} \frac{\partial\mathbf{h}_i}{\partial\mathbf{h}_{i-1}} = 
\text{diag} \left( g'(\mathbf{W}_{i}^{T} \mathbf{h}_{i-1} + \mathbf{b}_{i}) 
\right) \mathbf{W}_{i}.
```
with the non-linearity $`g(.)`$ and the corresponding element-wise gradient 
$`g'(.)`$. Using this approach the Jacobian is computed using a single feed-
forward pass and adds very limited computational complexity. The figure shows
the extended computational graph and the chaining of the partial derivatives for
the computation of the Jacobian.

<img src="/figures/differential_network.png" alt="drawing" width="800" align="middle"/>

## Prior Work:
This Deep Differential Network architecture was introduced in the ICLR 2019 
paper 
[Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning](https://arxiv.org/abs/1907.04490).
Within this paper the differential network was used to represent the kinetic and
potential energy of a rigid body. These energies as well as the Jacobian of the 
energies were embedded within the Euler-Lagrange differential equation and the
the physically plausible network parameters were learned by minimising
the residual of this differential equation on recored data. 

The deep differential network architecture was used in the following papers:

- [Lutter et. al., (2019). Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning,\
International Conference on Learning Representations (ICLR)](https://arxiv.org/abs/1907.04490).
- [Lutter et. al., (2019). Deep Lagrangian Networks for end-to-end learning of energy-based control for under-actuated systems,\
International Conference on Intelligent Robots & Systems (IROS)](https://arxiv.org/abs/1907.04489).
- [Lutter et. al., (2019). HJB Optimal Feedback Control with Deep Differential Value Functions and Action Constraints,\
Conference on Robot Learning](https://arxiv.org/abs/1909.06153)

If you use this implementation within your paper, please cite:

```
@inproceedings{lutter2019deep,
  author =      "Lutter, M. and  Ritter, C. and  Peters, J.",
  year =        "2019",
  title =       "Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning",
  booktitle =   "International Conference on Learning Representations (ICLR)",
}
```

## Example:
The example scripts ```1d_example_diff_net.py``` & ```2d_example_diff_net.py``` 
provide an example to train a differential network to approximate the 1d-function
$`f(x) = sin(x)`$ & the 2d-function $`f(x, y) = cos(x)sin(y)`$ and the respective 
Jacobian. The models can be either trained from scratch or the pre-trained 
models can be loaded from the directory folder. Below you can see the learned 
approximations of both functions and the Jacobian for the ReLu and Softplus 
activation. Right now the package supports Linear, ReLu, SoftPlus and Cosine 
activations but other non-linearities can be easily added.

#### 1d Example: $`f(x) = sin(x)`$
The **ReLu differential network** and the **SoftPlus differential network** are 
able to approximate the functions very accurately. However, only the SoftPlus 
network yields a smooth Jacobian, whereas the Jacobian of the ReLu network is 
piecewise constant due to the non-differentiable point of the ReLu activation.

Performance of the Different Models:

|   | Type 1 Loss / ReLu  | Type 1 Loss / SoftPlus  | Type 1 Loss / TanH  | Type 1 Loss / Cos |  Type 2 Loss / ReLu | Type 2 Loss / SoftPlus | Type 2 Loss / TanH | Type 2 Loss / Cos |
|---|---|---|---|---|---|---|---|---|
| MSE  $`f(\mathbf{x})`$                                    | 9.792e-6  | 6.909e-6  |  8.239e-7 | 1.957e-7  | 5.077e-7  | 8.666e-6  | 7.563e-6  | **1.746e-8**  |
| MSE  $`\partial f(\mathbf{x}) / \partial \mathbf{x}`$     | 4.514e-4  | 1.807e-4  |  3.099e-5 | 2.189e-6  | 2.661e-3  | 1.494e-4  | 4.345e-4  | **5.731e-7**  |
| MSE  $`\partial^2 f(\mathbf{x}) / \partial^2 \mathbf{x}`$ | 4.950e-1  | 5.930e-3  |  1.625e-3 | 5.451e-5  | 4.950e-1  | 4.519e-3  | 8.642e-3  | **1.209e-5**  |

<table border="5" bgcolor="white">
  <tr>
      <td align="center"><b>ReLu Differential Network</b></td>
      <td align="center"><b>SoftPlus Differential Network</b></td>
  </tr>
  <tr>
     <td align="center"><img src="/figures/1d_ReLu_diff_net.png" alt="drawing" width="500" align="center"/></td>
     <td align="center"><img src="/figures/1d_SoftPlus_diff_net.png" alt="drawing" width="500" align="center"/></td>
  </tr>
  <tr>
      <td align="center"><b>TanH Differential Network</b></td>
      <td align="center"><b>Cosine Differential Network</b></td>
  </tr>
  <tr>
     <td align="center"><img src="/figures/1d_Tanh_diff_net.png" alt="drawing" width="500" align="center"/></td>
     <td align="center"><img src="/figures/1d_Cos_diff_net.png" alt="drawing" width="500" align="center"/></td>
  </tr>
</table>


#### 2d Example: $`f(x, y) = cos(x)sin(y)`$
The **ReLu differential network** and the **SoftPlus differential network** are 
able to approximate the functions very accurately. However, only the SoftPlus 
network yields a smooth Jacobian, whereas the Jacobian of the ReLu network is 
piecewise constant due to the non-differentiable point of the ReLu activation.

Performance of the Different Models:

|   | Type 1 Loss / ReLu  | Type 1 Loss / SoftPlus  | Type 1 Loss / TanH  | Type 1 Loss / Cos |  Type 2 Loss / ReLu | Type 2 Loss / SoftPlus | Type 2 Loss / TanH | Type 2 Loss / Cos |
|---|---|---|---|---|---|---|---|---|
| MSE  $`f(\mathbf{x})`$                                     | 4.151e-5  | 2.851e-6  | 2.583e-6  | **2.929e-7**  | 3.293e-5  | 4.282e-5  | 8.361e-6  | 2.118e-6  |
| MSE  $`\partial f(\mathbf{x}) / \partial \mathbf{x}`$      | 2.014e-3  | 3.656e-5  | 1.525e-5  | **4.344e-6**  | 4.962e-3  | 8.128e-4  | 4.349e-4  | 7.038e-5  |
| MSE  $`\partial^2 f(\mathbf{x}) / \partial^2 \mathbf{x}`$  | 9.996e-1  | 1.288e-3  | 8.535e-4  | **1.709e-4**  | 9.996e-1  | 1.468e-2  | 1.163e-2  | 1.316e-3  |


<table border="5" bgcolor="white">
  <tr>
      <td align="center"><b>ReLu Differential Network</b></td>
  </tr>
  <tr>
     <td align="center"><img src="/figures/2d_ReLu_diff_net.png" alt="drawing" width="900" align="center"/></td>
  </tr>
  <tr>
      <td align="center"><b>SoftPlus Differential Network</b></td>
  </tr>
  <tr>
     <td align="center"><img src="/figures/2d_SoftPlus_diff_net.png" alt="drawing" width="900" align="center"/></td>
  </tr>
  <tr>
      <td align="center"><b>TanH Differential Network</b></td>
  </tr>
  <tr>
     <td align="center"><img src="/figures/2d_Tanh_diff_net.png" alt="drawing" width="900" align="center"/> </td>
  </tr>
  <tr>
      <td align="center"><b>Cosine Differential Network</b></td>
  </tr>
  <tr>
     <td align="center"><img src="/figures/2d_Cos_diff_net.png" alt="drawing" width="900" align="center"/> </td>
  </tr>
</table>


## Installation:
For installation this python package can be cloned and installed via pip
```
git clone https://github.com/milutter/deep_differential_network.git deep_differential_network
pip install deep_differential_network
```

## Contact:
If you have any further questions or suggestions, feel free to reach out to me via
```michael AT robot-learning DOT de```

