### Matrix Multiplication (Matmul)

You can perform efficient matrix multiplications by leveraging the hardware-accelerated **dot product instructions** introduced in this architecture[cite: 63]. These instructions are fundamental to high-performance `matmul` kernels, especially for AI and machine learning workloads.

The key instructions are `V_DOT*` operations, which operate on packed data types like 16-bit floats (`F16`), 8-bit integers (`I8`), or even 4-bit integers (`I4`)[cite: 64, 65, 66, 67].

Here's the general approach for a `matmul` ($C = A \\times B$):

1.  **Initialization**: Each work-item is responsible for calculating one or more elements of the output matrix C. The accumulator VGPR for the final result is initialized to zero.
2.  **Main Loop**: Loop over the K-dimension of the input matrices.
      * **Load Data**: Use vector memory instructions like `BUFFER_LOAD_DWORD` to load a vector from matrix A (a row) and a vector from matrix B (a column) into VGPRs[cite: 594].
      * **Compute Dot Product**: Use a `V_DOT*` instruction to compute the dot product of the loaded vectors and add the result to the accumulator VGPR[cite: 1459]. For example, `V_DOT2_F32_F16` calculates `D.f32 = S0.f16[0] * S1.f16[0] + S0.f16[1] * S1.f16[1] + S2.f32`, where `S2` is the accumulator[cite: 1459].
      * **Sync**: Use `S_WAITCNT` to ensure the data loads have completed before they are used by the dot product instruction[cite: 280, 1363].
3.  **Store Result**: After the loop finishes, the accumulator VGPR holds the final value for an element in matrix C. Use a vector memory instruction like `BUFFER_STORE_DWORD` to write this value to memory[cite: 594].

**Example `matmul` kernel pseudo-code:**

```c
// Each work-item computes one element C[y][x]
// SGPRs hold base addresses for A, B, C and the matrix dimension K
// VGPRs hold the work-item's x/y indices

v_mov_b32 v_acc, 0.0          // Initialize accumulator to zero

s_mov_b32 s_loop_count, K     // Initialize loop counter

loop:
    // Load 4 elements from A and B using VGPR addresses
    buffer_load_dwordx2 v_A_data, ...
    buffer_load_dwordx2 v_B_data, ...

    s_waitcnt vmcnt(0)          // Wait for loads to complete

    // Assumes A and B have F16 data. This performs 4 FMAs.
    v_dot4_i32_i8 v_acc, v_A_data, v_B_data, v_acc // Accumulate dot product

    s_sub_i32 s_loop_count, s_loop_count, 1
    s_cbranch_scc1 loop         // Branch if loop is not done

// Store final result
buffer_store_dword v_acc, ...
```

-----

### Other Fancy Operations 🚀

The "Vega" 7nm ISA includes several other powerful instructions for specialized, high-performance tasks.

#### Packed Math (SIMD within a Lane)

The `VOP3P` microcode format supports **packed math**, allowing you to perform two 16-bit operations in parallel within a single 32-bit VGPR[cite: 453, 516]. This is extremely useful for increasing throughput on smaller data types.

  * `V_PK_ADD_F16`: Adds two pairs of 16-bit floats simultaneously[cite: 51, 1457].
  * `V_PK_MAD_I16`: Performs two 16-bit integer multiply-adds in parallel[cite: 44, 1457].
  * `V_PK_FMA_F16`: A fused multiply-add for two pairs of 16-bit floats[cite: 51, 1457, 1517].

#### Wavefront Lane Shuffling

You can perform complex data shuffling between the 64 work-items in a wavefront without needing to use memory. These instructions use the LDS hardware for an arbitrary inter-lane swizzle. This is great for algorithms like FFTs, transpositions, or reductions.

  * **`DS_SWIZZLE_B32`**: Provides a variety of fixed swizzle patterns, including specialized modes for FFTs and rotations[cite: 1254, 1522].
  * **`DS_PERMUTE_B32` (Forward)**: Each work-item writes its data to a destination lane specified by its address VGPR. This is a "scatter" type operation[cite: 1508].
  * **`DS_BPERMUTE_B32` (Backward)**: Each work-item reads data from a source lane specified by its address VGPR. This is a "gather" type operation and supports broadcasting (multiple lanes reading from the same source)[cite: 1509].

#### Image & Video Processing

The ISA includes instructions that accelerate common computer vision and video encoding tasks.

  * **Sum of Absolute Differences (SAD)**: These instructions calculate the sum of absolute differences between vectors, which is a core operation in motion estimation.
      * `V_SAD_U8`: Calculates SAD on four packed 8-bit unsigned integers and adds the result to a 32-bit accumulator[cite: 1472].
      * `V_QSAD_PK_U16_U8`: Quad-SAD on packed 8-bit integers, accumulating into two 16-bit results[cite: 1485].
  * **Byte Permute**:
      * `V_PERM_B32`: Performs a byte-level permutation on two 32-bit source VGPRs based on a selector in a third VGPR, allowing for flexible rearrangement of bytes within a Dword[cite: 1484].

#### Specialized Math Helpers

For complex mathematical functions, there are hardware helpers to accelerate the most difficult parts.

  * **Trigonometric Pre-Op**: `V_TRIG_PREOP_F64` is a specialized instruction for high-precision trigonometric functions. It performs a lookup of 2/π to assist in the range reduction of large arguments for functions like `sin` and `cos`[cite: 1499, 1500].
  * **Division Helpers**: Division is often implemented with a reciprocal approximation followed by Newton-Raphson iterations. These instructions help handle the tricky parts.
      * `V_DIV_SCALE_*`: Pre-scales the numerator or denominator to avoid subnormal intermediate values that would lose precision[cite: 1478, 1480].
      * `V_DIV_FIXUP_*`: Detects and corrects for special cases like division by zero or infinity after the main calculation is done[cite: 1474, 1476].