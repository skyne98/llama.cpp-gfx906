Here is a developer reference cheatsheet for the AMD "Vega" 7nm ISA, focusing on its application in Machine Learning and AI.

### Architecture for Machine Learning

The "Vega" 7nm GCN architecture is designed for high-throughput parallel computation, making it well-suited for ML workloads. In an ML context, a **work-item** can be thought of as a processing element handling a single point in a tensor, while a **wavefront** is a group of 64 such elements executing a kernel in lockstep (SIMD).

* [cite_start]**Scalar vs. Vector Units**: The **SALU** is used for control flow (looping over tensor dimensions) and managing pointers, while the **VALU** performs the parallel mathematical operations on tensor data[cite: 137, 141].
* **Memory Hierarchy**:
    * [cite_start]**Global Memory**: Stores large datasets, model weights, and activations[cite: 176].
    * **LDS (Local Data Share)**: A 64 kB, high-bandwidth scratchpad memory essential for performance. [cite_start]It's used for **tiling** (blocking) strategies in `matmul` and convolutions, allowing a work-group to cache frequently reused data from global memory, drastically reducing latency[cite: 172, 1200].
    * [cite_start]**SGPRs/VGPRs**: Scalar registers hold uniform data like base pointers and dimension sizes, while Vector registers hold the unique data for each element being processed[cite: 184].

---

### Key Hardware Features for AI/ML Acceleration

This ISA includes specialized features that directly accelerate common ML operations.

#### Packed Math and Dot Product Acceleration

[cite_start]The most significant features for ML are the hardware-accelerated **dot product** and **packed math** instructions[cite: 42, 63, 64, 65, 66, 67]. These are crucial for the multiply-accumulate operations that dominate convolutions and matrix multiplications.

* [cite_start]**Mixed Precision**: These instructions natively support low-precision data types common in AI inference, such as 16-bit floats (`F16`), 8-bit integers (`I8`), and even 4-bit integers (`I4`), while often using a 32-bit accumulator for higher precision[cite: 64, 65, 66, 67, 1457].
* **High Throughput**: By packing smaller data types into 32-bit registers, these instructions perform multiple operations per clock cycle per work-item, significantly increasing computational throughput. [cite_start]For instance, `V_DOT4_I32_I8` performs four `I8` multiply-adds in a single instruction[cite: 1545].
* [cite_start]**Fused Operations**: Packed instructions like `V_PK_FMA_F16` perform a fused multiply-add on two pairs of 16-bit floats simultaneously, improving speed and precision[cite: 51, 1457].

#### Wavefront and Data Share Operations

Efficient data movement is critical. The ISA provides powerful tools for inter-thread communication and data rearrangement.

* [cite_start]**Wavefront Lane Shuffling**: The `DS_PERMUTE_B32` and `DS_BPERMUTE_B32` instructions use the LDS hardware to perform arbitrary data swaps ("swizzles") between the 64 lanes of a wavefront without writing to memory[cite: 1508, 1509]. This is ideal for high-performance reduction operations (e.g., `ReduceSum`, `ReduceMax`).
* [cite_start]**LDS Atomics**: Instructions like `DS_ADD_U32` and `DS_MAX_F32` perform atomic read-modify-write operations directly in the LDS[cite: 1472, 1473]. This is essential for accumulating partial results from multiple wavefronts in a work-group without race conditions.

---

### Mapping ML Kernels to the ISA

Here’s how to implement core ML operations using "Vega" 7nm instructions.

#### Matrix Multiplication & Convolution

These operations are fundamentally composed of dot products. A high-performance kernel uses a **tiling** strategy with the LDS.

1.  [cite_start]**Tiling**: A work-group loads small tiles of the input matrices/tensors from global memory into the LDS using `BUFFER_LOAD_*` instructions[cite: 1525]. This allows for data reuse, as each value loaded into the LDS will be used in multiple calculations.
2.  **Computation**: Within the work-group, each wavefront processes its portion of the tile.
    * Work-items loop through the K-dimension of the tiles stored in LDS.
    * [cite_start]In each iteration, they use a **`V_DOT*`** instruction (e.g., `V_DOT4_I32_I8`) to compute a partial sum, accumulating the result in a VGPR[cite: 1545].
3.  [cite_start]**Synchronization**: `S_BARRIER` is used to ensure all work-items in the work-group have finished loading a tile into LDS before computation begins, and finished computing with the current tile before loading the next one[cite: 279]. [cite_start]`S_WAITCNT vmcnt(0)` is used to ensure memory loads complete before the data is used[cite: 280, 282].
4.  [cite_start]**Store Output**: Once all tiles have been processed, the final accumulated results are written from VGPRs to the output tensor in global memory using `BUFFER_STORE_*` instructions[cite: 1525].

#### Element-wise Operations & Activation Functions

These operations map directly to standard VALU instructions, applied per-element.

* [cite_start]**Bias Adds / Residual Connections**: Use `V_ADD_F32` or `V_ADD_F16`[cite: 486, 490].
* [cite_start]**ReLU Activation**: Implemented with `V_MAX_F32` or `V_MAX_F16` (e.g., `v_max_f32 v_out, 0.0, v_in`)[cite: 486, 490].
* [cite_start]**Complex Activations (Sigmoid, Tanh)**: Composed from basic building blocks like `V_EXP_F32` and `V_RCP_F32`[cite: 1405].

#### Reduction Operations (e.g., Global Average Pooling)

Reductions are typically a multi-step process.

1.  **Intra-Wavefront Reduction**: Each wavefront reduces its 64 values down to a single value. [cite_start]This is done efficiently using `DS_PERMUTE_B32` to perform a parallel tree reduction (e.g., swapping and adding values from lanes that are 32, 16, 8, 4, 2, and 1 apart)[cite: 1508].
2.  **Inter-Wavefront Reduction**: The single result from each wavefront is written to a designated area in the LDS. [cite_start]An atomic instruction like `DS_ADD_RTN_U32` is used to safely accumulate the results from all wavefronts in the work-group[cite: 1472]. One thread then reads the final sum from the LDS.

#### Quantization & Data Type Conversion

Converting between high-precision (`FP32`) and low-precision (`FP16`/`INT8`) formats is essential for optimizing inference performance.

* [cite_start]**Conversion**: The `V_CVT_*` family of instructions handles data type conversions (e.g., `V_CVT_F16_F32`, `V_CVT_I32_F32`)[cite: 1399].
* [cite_start]**Packing**: Use instructions like `V_CVT_PKNORM_I16_F32` to convert two 32-bit floats to two 16-bit normalized integers and pack them into a single 32-bit register, which is highly efficient for memory operations[cite: 1492].