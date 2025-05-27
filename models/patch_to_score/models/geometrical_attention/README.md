## Input
### features
F : (num_proteins, 10, 9) <br>
-> broadcast num_patches (amount of actual patches in the protein), num_amino_acids (total amount of amino acis in protein, regardless of patches) <br>
F : (num_proteins, 10, 11)

### coordinates
C : (num_proteins, 10, 3) <br>
-> distance matrix (L2 between patches) <br>
-> notice that this operation is rotation and translation invariant. <br> 
D : (num_proteins, 10, 10)

## Operation
We can now have a graph, where every node has a vector of features, and there are distances between the nodes.

Optional transitions:
1. {d_ij} <br>
d_ij = gauss_kernel(d_ij) (optional multi gaussians)
2. {d_ij} <br>
d_ij = Linear(d_ij) : adds dimension (num_proteins, 10, 10, c_d) <br>
But might be strong enough with just c_d = 1

### PatchesAttentionBlock({f_i}, {d_ij}, c_f = 32, h = 8)
f_i = LayerNorm(f_i) <br>
q_ih, k_ih, v_ih = LinearNoBias(f_i) <br>
b_ijh = LinearNoBias(LayerNorm(d_ij)) : This is c_d -> h <br> 
a_ijh = 1/sqrt(c_f) * softmax_j(q_ih_T * k_ih + b_ijh) <br>
o_ih = sigma_j (a_ijh * v_jh) <br>
f_i = Linear(Concat_h(o_ih))

