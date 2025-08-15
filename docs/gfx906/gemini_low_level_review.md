

# **A Low-Level Programmer's Guide to the AMD GFX906 (Instinct MI50) Architecture**

## **Section 1: The GFX9 (Vega) Architectural Foundation**

The AMD Instinct MI50 accelerator, identified by the hardware architecture name gfx906, represents a significant milestone in the evolution of GPU computing. To program this hardware at a low level, a foundational understanding of its underlying microarchitecture is not merely beneficial but essential. The MI50 is built upon the "Vega 20" GPU, which is a 7nm die shrink and enhancement of the "Vega 10" design.1 Both are implementations of the Graphics Core Next (GCN) 5.1 microarchitecture, more commonly known as "Vega".3 This architecture was not an incremental update; it was, as described by AMD, the most sweeping change to its core graphics technology since the introduction of the first GCN-based chips.5 For the low-level programmer, this translates to a new set of capabilities and a fundamentally different approach to memory management and command processing compared to prior generations.

### **1.1. The GCN 5.1 "Vega" Microarchitecture: A Sweeping Change**

The Graphics Core Next (GCN) architecture is the bedrock of AMD's GPU designs from 2012 through the Vega generation. It is a scalar-vector design that organizes computation into a hierarchical structure. At the highest level, the GPU is composed of one or more Shader Engines (or Shader Arrays). These arrays contain a collection of Compute Units (CUs), which are the fundamental processing blocks of the GCN architecture.3

Each CU in the Vega architecture is a potent computational engine. It contains four SIMD (Single Instruction, Multiple Data) Vector Units, each 16 lanes wide, a scalar unit with its own ALU, a dedicated instruction buffer and scheduler, a 64 KiB Local Data Share (LDS) for fast scratchpad memory, and L1 cache.5 Work is dispatched to the CUs in the form of "wavefronts," which are groups of 64 threads (often called "work-items" or "lanes") that execute in a SIMD fashion. While all 64 threads in a wavefront execute the same instruction at any given time (lockstep execution), an execution mask allows individual threads to be deactivated, enabling divergent control flow within a wavefront.6

The Instinct MI50, as an implementation of the Vega 20 GPU, is specifically designated by the target ID gfx906 in the AMD software ecosystem, particularly within the LLVM compiler toolchain.7 This identifier is crucial, as it signals to the compiler to generate machine code that leverages the specific instruction set extensions and adheres to the hardware characteristics of this particular chip.

### **1.2. Command Processing and Scheduling: The GPU's Front Door**

The execution of any workload on the GPU begins at the command processing stage. The Vega architecture features a sophisticated front-end designed to efficiently fetch, decode, and schedule work from multiple independent sources. This front-end comprises two main types of hardware units: the Graphics Command Processor (GCP) and the Asynchronous Compute Engines (ACEs).3

The GCP is primarily responsible for handling graphics command streams, managing the traditional graphics pipeline for rendering tasks. The ACEs, in contrast, are dedicated to processing compute workloads. Each ACE can manage multiple independent command queues, allowing the GPU to interleave and execute tasks from different applications or different streams within the same application concurrently.3 This capability is the hardware foundation for "Asynchronous Compute," a key feature of GCN that allows the GPU to utilize idle resources by running compute tasks (e.g., physics simulations, post-processing) in the gaps left by graphics workloads that might be bottlenecked by fixed-function hardware or memory bandwidth.3

The command submission model involves the host CPU (via the kernel driver or a user-space runtime) writing command packets into one or more command queues residing in system memory. The GCP and ACEs then fetch these packets, decode them, and dispatch the work to the CUs.3

This process is managed by a two-tiered hardware scheduling system. A high-level scheduler, sometimes referred to as the "workload manager," is responsible for scheduling the execution of entire draw and compute queues. It makes strategic decisions about when to execute compute operations to fill underutilized CUs.3 Once a command (e.g., a kernel launch) is dispatched to the CUs, a lower-level CU Scheduler takes over. This scheduler manages the execution of individual wavefronts within the CU, deciding which wavefront to issue an instruction from next, hiding memory latency by swapping between active wavefronts, and managing the flow of data through the CU's pipelines.3 For a low-level programmer, understanding this dual-level scheduling is key to structuring workloads that keep the hardware's deep pipelines fully saturated.

### **1.3. The Vega Memory Subsystem: A Paradigm Shift**

Perhaps the most revolutionary aspect of the Vega architecture is its completely redesigned memory subsystem. This subsystem is built around two core technologies: second-generation High-Bandwidth Memory (HBM2) and the High-Bandwidth Cache Controller (HBCC).5

The Instinct MI50 utilizes HBM2, a type of stacked DRAM that is co-packaged with the GPU on a silicon interposer. This provides an extremely wide memory interface, resulting in memory bandwidth that is an order of magnitude higher than traditional GDDR memory. This vast bandwidth is critical for feeding the thousands of parallel threads in the CUs, especially for the memory-intensive workloads common in high-performance computing (HPC) and AI.4

The true paradigm shift, however, comes from the HBCC. In previous GPU architectures, the GPU's local video memory (VRAM) was a distinct memory space. Data had to be explicitly copied by the programmer from host system memory into VRAM before the GPU could access it. This explicit memory management was a major source of programming complexity and a frequent performance bottleneck.5 The HBCC fundamentally alters this model. It transforms the GPU's local HBM2 into a last-level cache for a vastly larger, unified virtual address space. The Vega architecture supports a 49-bit virtual address space, allowing it to address up to 512 TB of memory.5 This virtual address space can encompass not only the local HBM2 but also system RAM and, in some configurations, even non-volatile storage like SSDs.

When a kernel attempts to access an address in this virtual space, the HBCC handles the translation. If the data is already present in the HBM2 (a cache hit), access is fast. If the data is not present (a cache miss), the HBCC will automatically issue a request over the PCIe bus or Infinity Fabric to fetch the required memory page from system RAM and place it into the HBM2, evicting another page if necessary.5 This hardware-managed caching mechanism liberates the programmer from the need to perform manual

memcpy operations between host and device.

This architectural change has profound implications for low-level programming. While it simplifies memory management by creating a unified pointer space, it shifts the focus of performance optimization. Instead of managing explicit data transfers, the programmer must now focus on data locality. The performance difference between an HBCC cache hit (accessing local HBM2) and a cache miss (stalling while a page is fetched from system memory) is immense. Therefore, efficient low-level programming on Vega requires structuring algorithms and data layouts to maximize temporal and spatial locality, ensuring that the working set of data remains resident in the HBM2 cache as much as possible.

The full memory hierarchy available to a single work-item is thus:

1. **Private Vector General-Purpose Registers (VGPRs):** The fastest memory, private to each thread.  
2. **Local Data Share (LDS):** A 64 KiB software-managed scratchpad, shared by all threads within a work-group executing on a single CU. It is essential for low-latency inter-thread communication.6  
3. **L1 Caches:** Each CU has L1 caches for vector and scalar data.10  
4. **L2 Cache:** A large L2 cache (4 MB on Vega 10\) is shared by all CUs, serving as a backstop for the L1 caches.5  
5. **HBM2 (High-Bandwidth Cache):** The local on-package memory, managed by the HBCC.  
6. **System Memory:** Off-chip DRAM accessible via the PCIe bus or Infinity Fabric, transparently managed by the HBCC.

Additionally, the architecture includes a 64 KiB Global Data Share (GDS), a small scratchpad memory that is accessible by all CUs across the entire GPU. While its small size limits its general-purpose use, it can be valuable for specific algorithms that require fast, low-latency communication or atomic operations across different work-groups.6

### **1.4. Infinity Fabric: The Coherent Backbone**

Tying the entire Vega architecture together is the Infinity Fabric. Vega was the first AMD GPU to incorporate this high-speed, low-latency, coherent interconnect, which was co-developed for and shared with AMD's "Zen" family of CPUs.4

Infinity Fabric acts as the central nervous system of the SoC-style chip design. It connects all the major IP blocks on the die: the graphics core (the CUs), the memory controllers for the HBM2, the HBCC, the PCIe controller, the display engine, and the video acceleration blocks.5 Its key feature is coherency, which means it provides a protocol for ensuring that all agents on the fabric have a consistent view of memory. This is a critical enabling technology for features like the HBCC, which needs to maintain coherence between the L2 cache and the data stored in system memory.

The adoption of a standardized, modular interconnect like Infinity Fabric allows for a more flexible approach to chip design. It also lays the groundwork for tighter integration between CPUs and GPUs in future APUs and multi-chip-module designs, pushing the industry further toward truly heterogeneous systems.5 For the Instinct MI50, the Infinity Fabric provides the high-bandwidth, low-latency pathway necessary for the HBCC to efficiently service page faults from system memory, making the unified virtual memory model a practical reality.

## **Section 2: The GFX9 Instruction Set Architecture (ISA)**

A direct command of the Instruction Set Architecture (ISA) is the ultimate goal of any low-level programming endeavor. The AMD GFX9 architecture, also known as GCN 5.1, features a rich and complex ISA designed for massively parallel computation. For the programmer targeting the Instinct MI50 (gfx906), a precise understanding of this instruction set is paramount. However, the path to this understanding is not straightforward, as the necessary information is spread across multiple sources of varying age, format, and authority.

### **2.1. The Documentation Dichotomy: Official PDFs vs. LLVM's Living Record**

Navigating the documentation for the GFX9 ISA requires a dual-pronged approach, leveraging both official architectural manuals and the source code of the primary compiler toolchain.

**Official AMD ISA Documents:** AMD has a history of publishing detailed PDF documents for its GPU ISAs. For the Vega architecture, the key document is the "AMD ‘Vega’ Instruction Set Architecture Reference Guide".6 This document is an invaluable resource for understanding the high-level concepts of the architecture. It provides detailed descriptions of the programming model, the organization of program state (registers, memory spaces), the memory model, and the intended operational semantics of the instruction families. It explains the "what" and "why" behind the architecture's design. However, these documents have limitations: they are static snapshots in time and may not be updated to reflect hardware errata discovered after publication. Furthermore, while they describe instruction behavior, they do not always provide the exact, literal syntax required by an assembler.

**The LLVM amdgcn Backend as Ground Truth:** For practical, hands-on programming, the most accurate and authoritative source of ISA information is the AMDGPU backend within the open-source LLVM compiler project.7 The ROCm software stack, which is AMD's official platform for GPU computing, uses a

clang/LLVM-based compiler to generate the final machine code that runs on the hardware.17 Consequently, the representation of the ISA within this compiler—its instruction mnemonics, operand syntax, available modifiers, and binary encodings—is, by definition, correct and functional. It is the living record of what the hardware actually accepts. This makes browsing the LLVM source code, particularly the target description files (

.td) and assembler parsers, an essential activity for any serious low-level developer.

This compiler-as-specification approach is more than just a matter of convenience; it is a necessity for correctness. The LLVM source code is the only public repository for information on certain hardware bugs and the compiler workarounds implemented to avoid them. These are often defined as SubtargetFeature flags within the AMDGPU.td file.18 For a programmer writing assembly by hand, being unaware of these errata can lead to generating code that, while syntactically valid, triggers a hardware flaw, resulting in silent data corruption or system hangs. Therefore, the LLVM source code must be treated as the de facto ISA specification, providing a level of detail and real-world accuracy that static PDF documents cannot match.

For more recent architectures like RDNA and CDNA, AMD has begun providing machine-readable ISA specifications in XML format, along with a C++ IsaDecoder API to parse them.19 While GFX9 is not a primary target of this modern initiative, it signals a broader trend in the industry to move documentation closer to the code, further reinforcing the idea of the toolchain as the ultimate source of truth.

### **2.2. Instruction Categories and Formats**

The GFX9 ISA is divided into several categories based on the hardware unit that executes them and the number of operands they take. The syntax presented here is derived from the LLVM amdgcn backend documentation.12

**Scalar Operations (SOP):** These instructions are executed by the scalar unit and operate on the Scalar General-Purpose Registers (SGPRs), which are shared by all 64 threads in a wavefront.

* SOP1: Scalar operations with one source operand. Examples: s\_mov\_b32 s0, s1 (move), s\_not\_b32 s0, s1 (bitwise NOT).  
* SOP2: Scalar operations with two source operands. Examples: s\_add\_i32 s0, s1, s2 (integer add), s\_and\_b32 s0, s1, s2 (bitwise AND).  
* SOPC: Scalar comparison operations. These operations compare two scalar operands and write a single bit result to the Scalar Condition Code (SCC) register. Example: s\_cmp\_eq\_i32 s0, s1 (compare equal).  
* SOPK: Scalar operations with a signed 16-bit immediate constant (simm16). These are used for operations involving small constants. Example: s\_movk\_i32 s0, 0x1234.  
* SOPP: Scalar operations for program control. This is a critical category that includes branches, waits, and program termination. Examples: s\_branch \<label\>, s\_cbranch\_scc0 \<label\> (conditional branch on SCC), s\_waitcnt vmcnt(0) (wait for vector memory operations), s\_endpgm (end program).

**Vector ALU Operations (VOP):** These instructions are executed by the SIMD units and operate on the Vector General-Purpose Registers (VGPRs). Each of the 64 threads in a wavefront has its own private set of VGPRs, and a single VOP instruction performs the same operation on the corresponding VGPRs for all active threads in parallel.

* VOP1: Vector operations with one source operand. Examples: v\_mov\_b32 v0, v1, v\_cvt\_f32\_f16 v0, v1 (convert 16-bit float to 32-bit float).  
* VOP2: Vector operations with two source operands. Examples: v\_add\_f32 v0, v1, v2, v\_mul\_f32 v0, v1, v2.  
* VOP3: Vector operations with three source operands. This format is common for fused operations like Fused Multiply-Add (FMA), which calculates (src0 \* src1) \+ src2. Example: v\_fma\_f32 v0, v1, v2, v3.  
* VOPC: Vector comparison operations. These compare two vector operands on a per-lane basis and write the 64-bit result mask to the Vector Condition Code (VCC) register. Example: v\_cmp\_eq\_f32 vcc, v0, v1.  
* VOP3P: Packed vector operations. These instructions perform operations on packed data types (e.g., two 16-bit values packed into a single 32-bit register), which is a key feature for accelerating mixed-precision workloads.12

**Vector Memory Operations:** These instructions are responsible for moving data between VGPRs and memory.

* FLAT: These are the primary memory access instructions in the Vega architecture. They operate on the unified virtual address space provided by the HBCC, allowing them to access global memory, scratch (private) memory, or LDS memory with a single instruction type.12 Examples:  
  flat\_load\_dword v0, v\[1:2\], flat\_store\_dword v\[1:2\], v0, flat\_atomic\_add v0, v\[1:2\], v3.  
* MUBUF: Untyped Buffer memory instructions. These are used to access memory through a buffer resource descriptor, which provides information about the memory region's base address and size.  
* MIMG: Image Memory instructions. These are specialized instructions for accessing texture and image data, supporting operations like sampling with filtering.  
* MTBUF: Typed Buffer memory instructions. These are similar to MUBUF but interpret the data according to a specific format.

**Data Share (DS) and Scalar Memory (SMEM):**

* DS: Instructions for accessing the on-chip Local Data Share (LDS). These are highly optimized for low-latency communication between threads within the same work-group. Examples: ds\_read\_b32 v0, v1, ds\_write\_b32 v1, v0, ds\_add\_u32 v1, v0.  
* SMEM: Instructions for the scalar unit to read from memory. These are typically used to load constant data or buffer descriptors that are uniform across the entire wavefront. Example: s\_load\_dword s0, s\[4:5\], 0x0.

### **2.3. GFX906-Specific Instructions: The AI Accelerators**

The Instinct MI50 (gfx906) is not just a generic Vega GPU; it was specifically designed with features to accelerate the mathematical operations at the heart of machine learning and AI workloads. These features manifest as a set of new instructions, documented in the gfx906 target definition within LLVM, that are not present on the base gfx900 (Vega 10\) architecture.7

The most significant additions are instructions for high-throughput packed math and dot products. Deep learning models rely heavily on matrix multiplications, which can be decomposed into a vast number of dot products. The gfx906 ISA includes instructions that can compute these dot products on lower-precision integer or floating-point data at a much higher rate than standard 32-bit floating-point operations.

* v\_dot2\_f32\_f16 v0, v1, v2, v3: This instruction takes two source registers (v1, v2), each containing two packed 16-bit floating-point values. It computes the dot product of these two 2-element vectors and adds the result to a 32-bit float accumulator (v3), storing the final 32-bit result in v0.  
* v\_dot4\_i32\_i8 v0, v1, v2, v3: This performs a dot product on two 4-element vectors of 8-bit signed integers, accumulating the result into a 32-bit integer.  
* v\_dot8\_i32\_u4 v0, v1, v2, v3: This instruction further increases throughput by performing a dot product on two 8-element vectors of 4-bit unsigned integers.

These instructions are critical for accelerating inference workloads, where models are often quantized to lower-precision integers (INT8, INT4) to reduce memory footprint and increase computational throughput.

Additionally, gfx906 introduces instructions for mixed-precision Fused Multiply-Add (FMA) operations, such as v\_fma\_mix\_f32 and v\_fma\_mixlo\_f16.7 These allow FMA operations to be performed on operands of different precisions (e.g., multiplying two 16-bit floats and adding the result to a 32-bit float accumulator) within a single instruction. This is a common pattern in AI training algorithms that use mixed precision to balance performance and numerical stability.

### **2.4. Operands, Modifiers, and Encodings**

The expressiveness of the GFX9 ISA comes not just from its opcodes but from its rich set of operands and instruction modifiers. A comprehensive guide to the operand syntax is provided by the LLVM documentation.21

* **Registers:** The primary operands are registers. The ISA defines several register files:  
  * Scalar GPRs: s0 through s101 (or higher depending on configuration).  
  * Vector GPRs: v0 through v255.  
  * Special Registers: vcc (Vector Condition Code, a 64-bit mask), exec (Execution Mask, a 64-bit mask), m0 (a 32-bit register used for memory addressing and other temporary storage), and ttmp registers (a set of SGPRs reserved for trap handler use).  
* **Literals and Constants:** Instructions can often take immediate values as operands. These can be integer literals or special inline constants that represent commonly used floating-point values like 0.0, 1.0, 0.5, etc., which are encoded directly into the instruction word.  
* **Modifiers:** Many instructions can be customized with modifiers that alter their behavior without changing the opcode. Common modifiers include:  
  * clamp: When specified on a floating-point instruction, the result is clamped to the range \[0.0,1.0\].  
  * omod: Output modifiers that can be applied to the result of an instruction, such as multiplying by 2.0, 4.0, or 0.5.  
  * DPP (Data Parallel Primitives): A powerful set of modifiers for VOP instructions that enable efficient, low-latency data sharing between threads within a single wavefront, avoiding the need to use LDS memory.  
  * SDWA (Sub-DWORD Addressing): Modifiers that allow vector instructions to operate on smaller data types (e.g., bytes or half-floats) within a 32-bit VGPR without needing separate packed instructions.

### **2.5. Known Hardware Errata: The Undocumented Reality**

One of the most critical aspects of low-level programming is contending with the imperfections of the hardware itself. Silicon is not perfect, and chips often ship with minor design flaws, or errata, that can cause incorrect behavior under specific circumstances. Official documentation rarely, if ever, details these bugs. The only reliable public source for this information for AMD GPUs is often the LLVM target definition files (.td), which contain the compiler's implementation of workarounds.18

For the GFX9 architecture, the LLVM source code documents several such bugs that the compiler is programmed to avoid. These are typically represented as "features" that a specific GPU target either has or does not have. Key examples for GFX9 include 18:

* FeatureNegativeScratchOffsetBug: On GFX9, using a negative immediate offset in a scratch memory instruction (used for register spilling) could incorrectly cause a page fault. The compiler must implement a workaround, likely by avoiding the generation of such instructions.  
* FeatureOffset3fBug: A subtle hardware bug related to a specific branch offset value of 0x3f. The compiler must ensure it never generates a branch with this exact offset.  
* FeatureNSAtoVMEMBug: This bug describes a failure condition that can occur when a Non-Sequential Address (NSA) MIMG instruction is immediately followed by a standard VMEM (e.g., flat or buffer) instruction, but only when the exec mask is either all zeros in the low 32 bits or all zeros in the high 32 bits. The compiler must insert other instructions between these two to break the problematic pattern.

For a low-level programmer, this information is invaluable. Attempting to write GFX9 assembly without being aware of these issues is fraught with peril. A program might appear to work correctly most of the time but fail unpredictably when a specific data pattern or control flow path triggers one of these latent hardware bugs. This reinforces the necessity of treating the LLVM source code as the definitive reference, as it implicitly documents the "safe" subset of the ISA.

| Instruction Family | Description | Key Examples | GFX906 Specific? |
| :---- | :---- | :---- | :---- |
| **SOPP** | Scalar Program Flow Control | s\_branch, s\_cbranch\_scc0, s\_waitcnt, s\_endpgm | No |
| **SOPK** | Scalar Operation with Constant | s\_movk\_i32, s\_addk\_i32, s\_cmovk\_i32 | No |
| **SOP2** | 2-Operand Scalar ALU | s\_add\_u32, s\_and\_b64, s\_lshl\_b32 | No |
| **SOPC** | Scalar Compare | s\_cmp\_eq\_i32, s\_cmp\_lg\_u64 | No |
| **VOP2** | 2-Operand Vector ALU | v\_add\_f32, v\_mul\_i32\_i24, v\_and\_b32 | No |
| **VOPC** | Vector Compare | v\_cmp\_eq\_f32, v\_cmp\_lt\_u32 | No |
| **VOP3** | 3-Operand Vector ALU | v\_fma\_f32, v\_mad\_u32\_u24, v\_min3\_i32 | No |
| **DS** | Local Data Share Access | ds\_read\_b32, ds\_write\_b32, ds\_add\_rtn\_u32 | No |
| **FLAT** | Unified Virtual Memory Access | flat\_load\_dword, flat\_store\_dwordx2, flat\_atomic\_add | No |
| **SMEM** | Scalar Memory Read | s\_load\_dword, s\_buffer\_load\_dwordx4 | No |
| **VOP3P** | Packed Math for AI/ML | v\_dot2\_f32\_f16, v\_dot4\_i32\_i8, v\_fma\_mix\_f32 | **Yes** |

## **Section 3: The Hardware-Software Interface**

The Instruction Set Architecture defines the language of the hardware, but a program must also understand and manage the machine's state. This hardware-software interface encompasses the set of registers that define a wavefront's context, the rules governing memory consistency and ordering, and the initial state provided by the hardware when a kernel begins execution. Mastering this interface is the bridge between writing individual instructions and composing a correct, functional program.

### **3.1. The GFX9 Program State: Managing the Machine**

Each wavefront executing on a GFX9 CU maintains a specific set of architectural state, defined by a collection of special-purpose hardware registers. The official ISA manual provides a detailed account of this program state.6 A low-level program must read from and write to these registers to control its execution.

* **Program Counter (PC):** This is a 48-bit register that holds the byte address of the next instruction to be fetched for the wavefront. It is manipulated by program control instructions like s\_branch and s\_get\_pc.  
* **Execution Mask (exec):** This is a 64-bit register that is fundamental to the SIMD execution model of GCN. Each bit in the exec mask corresponds to one of the 64 threads (lanes) in the wavefront. For any given vector instruction, only the lanes with their corresponding bit set to 1 in the exec mask will execute the instruction and write back a result. Lanes with a bit of 0 are "masked off" and effectively perform a no-op. This mechanism is how the hardware handles divergent control flow (e.g., if/else blocks).  
* **Status Register (STATUS):** This is a 32-bit read-only register that provides a snapshot of the wavefront's current state. It contains a collection of single-bit flags, including:  
  * SCC: The current state of the Scalar Condition Code.  
  * EXECZ: A flag that is set to 1 if the exec mask is all zeros.  
  * VCCZ: A flag that is set to 1 if the VCC mask is all zeros.  
  * IN\_BARRIER: Indicates if the wavefront is currently waiting at a barrier.  
  * HALT: Indicates if the wavefront is in a halted state.  
* **Mode Register (MODE):** This is a 32-bit writable register that allows a program to configure certain aspects of the hardware's behavior. Key fields include:  
  * FP\_ROUND: Controls the rounding mode for floating-point operations (e.g., round to nearest even, round towards zero).  
  * FP\_DENORM: Controls how denormalized floating-point numbers are handled (e.g., flush to zero or preserve).  
  * IEEE: Enables strict IEEE-754 compliance for floating-point operations.  
  * EXCP\_EN: Enables or disables the generation of floating-point exception traps.  
* **Condition Code Registers (SCC and VCC):** These registers store the results of comparison operations and are used for conditional branching.  
  * SCC (Scalar Condition Code): A single bit that holds the boolean result of a scalar comparison instruction (SOPC). It is used by scalar conditional branch instructions like s\_cbranch\_scc0.  
  * VCC (Vector Condition Code): A 64-bit mask that holds the per-lane boolean results of a vector comparison instruction (VOPC). It can be used to update the exec mask, effectively selecting a subset of threads based on a condition.  
* **Trap and Exception Registers:** The architecture provides a set of registers for handling hardware exceptions, such as floating-point errors or memory access violations. These include TRAPSTS (Trap Status), TBA (Trap Base Address), TMA (Trap Memory Address), and a set of TTMP registers (Trap Temporary SGPRs) for use by the trap handler code.6

### **3.2. The GFX9 Memory Model: Rules for Coherency and Ordering**

A modern GPU is a massively parallel, memory-intensive system with a deep and complex memory hierarchy. To ensure correctness in the presence of thousands of concurrent memory operations, the hardware defines a strict memory consistency model. The LLVM documentation for the AMDGPU backend provides the most detailed public description of this model for GFX9.10

**Memory Scopes:** The model is defined in terms of memory scopes, which describe the visibility of memory operations to different groups of threads. The four key scopes are 10:

* **wavefront:** Operations are visible to other threads within the same wavefront.  
* **workgroup:** Operations are visible to all threads within the same work-group (which may be composed of multiple wavefronts). This is the scope of the LDS.  
* **agent:** Operations are visible to all threads running on the same GPU (the "agent").  
* **system:** Operations are visible to all agents in the system, including the CPU and other GPUs.

**Cache Hierarchy and Coherence:** The GFX9 memory model is characterized by its multiple levels of caching and specific coherence rules. Each CU has a vector L1 cache shared by its SIMDs. A separate scalar L1 cache is shared by a group of CUs. A crucial detail is that the vector L1 and scalar L1 caches are **not coherent** with each other.10 All CUs on the GPU share a unified L2 cache. While the L2 cache can be kept coherent with other system agents for certain memory types, the programmer must assume that, by default, caches on different CUs are not coherent.

This lack of automatic coherence means that if one CU writes to a memory location and another CU needs to read that data, the programmer must insert explicit instructions to ensure the data is written back from the first CU's caches to the L2 cache and that the second CU's caches are invalidated before the read.

**Synchronization Primitives:** The ISA provides instructions to enforce this ordering and visibility.

* **s\_waitcnt:** This is arguably the most critical instruction for ensuring correctness in any non-trivial GFX9 program. The hardware maintains several counters for in-flight operations, including vmcnt (outstanding vector memory operations), lgkmcnt (outstanding LDS, GDS, and scalar memory operations), and expcnt (outstanding export/GDS write operations). The s\_waitcnt instruction stalls the wavefront's execution until the specified counters have decremented to zero.10 For example,  
  s\_waitcnt vmcnt(0) forces the program to wait until all previously issued vector memory loads and stores have completed and their results are visible. This is essential for preventing read-after-write and write-after-write hazards between dependent memory operations.  
* **Memory Fences:** Instructions like s\_fence provide finer-grained control over memory ordering. They act as a barrier, ensuring that all memory operations of a certain type and scope issued before the fence are visible to other threads in that scope before any memory operations after the fence are executed.

A particularly subtle but critical aspect of the GFX9 memory model is the potential for reordering between LDS and vector memory operations. The LLVM documentation explains that because the LDS and the vector memory unit have separate request queues within the CU, operations issued by different wavefronts within the same work-group can have their visibility reordered.10 For instance, wavefront A might write to LDS, then write to global memory. Wavefront B, in the same work-group, might see the global memory write before it sees the LDS write. To prevent this, a

s\_waitcnt lgkmcnt(0) is required to ensure that all LDS operations are complete before subsequent vector memory operations from other wavefronts can be observed.

The centrality of s\_waitcnt cannot be overstated. In a highly parallel and out-of-order execution environment like a GPU, assumptions about program order translating directly to execution order are invalid. s\_waitcnt is not merely an optimization tool; it is a fundamental correctness primitive. For a low-level programmer, understanding where to insert these wait instructions is as critical as choosing the correct ALU instruction. Omitting a necessary s\_waitcnt will not result in slower code, but in unpredictable, non-deterministic data races that are nearly impossible to debug. The detailed explanation of the GFX9 memory model in the LLVM documentation is therefore one of the most valuable resources available, as it provides the rules needed to write correct code.

### **3.3. Initial Wavefront State and Kernel Launch**

When the Command Processor dispatches a kernel, the hardware automatically initializes the state of the first wavefront of each work-group. This initial state provides the kernel with its starting context, including its unique position within the compute grid and pointers to its arguments. The specific registers that are initialized are controlled by a set of enable\_\* bit-fields in the Kernel Descriptor data structure (which will be detailed in Section 4.4).10

**System SGPRs:** The hardware can pre-load a set of SGPRs with system-generated values. The compiler specifies which of these are needed via the kernel descriptor. The enabled registers are packed into the low-numbered SGPRs. Common system SGPRs include:

* Work-Group ID X, Y, Z: The 3D coordinate of the work-group within the dispatch grid.  
* Private Segment Buffer: A pointer to the scratch memory region for the wavefront.  
* Kernarg Segment Ptr: A pointer to the memory region containing the kernel's arguments.  
* Dispatch Ptr: A pointer to the dispatch packet.  
* Queue Ptr: A pointer to the AQL queue the dispatch originated from.

**User SGPRs:** In addition to system values, the first few SGPRs are typically used to pass kernel arguments directly. These are loaded by the hardware from the memory region pointed to by the Kernarg Segment Ptr.

**System VGPRs:** The hardware can also initialize the first few VGPRs for each thread with its unique Work-Item ID. The enable\_vgpr\_workitem\_id field in the kernel descriptor controls this. If set to 1, v0 is initialized with the work-item's X ID. If set to 2, v0 gets the X ID and v1 gets the Y ID, and so on.10 This saves the kernel from having to compute these values itself.

## **Section 4: The Path to Execution: Compiling and Packaging Kernels**

Writing instructions in assembly is only one part of the low-level programming process. To be executed, this code must be compiled into machine-readable binary, packaged into a standardized object format, and accompanied by critical metadata that describes its resource requirements to the hardware. This section details this toolchain and packaging pipeline, from the high-level software stack down to the bits and bytes of the final executable object.

### **4.1. The ROCm/HSA Software Stack: An Architectural Overview**

The AMD ROCm (Radeon Open Compute) platform is an open-source software stack designed for GPU computing. It provides the necessary components to bridge the gap between a user application and the GPU hardware. For the low-level programmer, it is essential to understand the layers of this stack, as each plays a distinct role in the execution pathway.17

* **High-Level Programming Models:** At the top of the stack are programming languages and APIs that provide abstractions for writing parallel code. The most prominent are HIP (Heterogeneous-Compute Interface for Portability), a C++-based model designed for easy porting of NVIDIA CUDA applications, and OpenCL, an open standard for heterogeneous computing.26 While a low-level programmer may choose to bypass these, they are built upon the layers below.  
* **Compiler Infrastructure:** ROCm uses a compiler based on Clang and LLVM. This compiler takes high-level code (like HIP C++) and lowers it through various intermediate representations until it finally generates GCN ISA machine code for a specific GPU target.17 This is the tool that produces the executable  
  .text section of a kernel.  
* **HSA (Heterogeneous System Architecture) Runtime:** The core of the user-space stack is the ROCR-Runtime, which implements the HSA Runtime API.29 This runtime is a library that provides the fundamental services an application needs to interact with the GPU. Its responsibilities include discovering available GPUs ("agents"), allocating memory that is visible to the GPU, creating command queues for work submission, and managing synchronization objects ("signals"). It is the direct interface to the kernel-mode driver.  
* **Kernel-Mode Driver (KMD):** At the lowest level is the amdgpu Linux kernel module, which is part of the ROCK-Kernel-Driver project.17 This privileged component is the only piece of software that communicates directly with the GPU's hardware registers. It manages device initialization, memory virtualization (GPUVM), interrupt handling, and power management. The HSA runtime communicates with the  
  amdgpu driver through a defined interface (ioctl calls) to request hardware resources like command queues.

### **4.2. The LLVM amdgcn Backend: The Toolchain**

The primary tool for compiling code for AMD GPUs is clang, the C/C++ frontend for the LLVM project. To target an AMD GPU, a specific target triple must be used: amdgcn-amd-amdhsa.10 This triple informs the compiler that it should generate code for the

amdgcn architecture, for a device from vendor amd, targeting the amdhsa (HSA) operating system/ABI.

The most critical compiler flag for a low-level programmer is \-mcpu. This flag specifies the exact GPU architecture to target. To generate code optimized for and compatible with the Instinct MI50, the programmer must specify \-mcpu=gfx906.10 Using this flag ensures that the compiler will:

1. Generate instructions from the correct GFX9 ISA variant, including the gfx906-specific packed math and dot product instructions.  
2. Apply workarounds for any known hardware errata specific to the gfx906 chip.  
3. Schedule instructions based on the latency and throughput characteristics of the gfx906 microarchitecture.

Recently, the LLVM project has begun adding support for "generic targets," such as gfx9-generic.32 The goal of these targets is to produce a single binary that can run on multiple different GPUs within the same family (e.g., both a Vega 10 and a Vega 20 GPU). This is achieved by generating code that only uses the common subset of instructions and may be less aggressively scheduled. While this offers portability, it comes at the cost of performance and the inability to use chip-specific features, making the explicit

\-mcpu=gfx906 flag the preferred choice for maximum performance on the MI50.

### **4.3. The HSA Code Object Format: The GPU's Executable**

Once the compiler generates the machine code, it must be packaged into a format that the HSA runtime and loader can understand. This format is a standard 64-bit ELF (Executable and Linkable Format) object file, with specific conventions for AMD GPUs.10 The full details of this format are specified in the AMDGPU-ABI document.28

The ELF header of an HSA code object is marked with ELFOSABI\_AMDGPU\_HSA in the e\_ident field, which unambiguously identifies it as a file intended for the HSA platform.13 The object file contains several key sections:

* .text: This section contains the raw binary machine code for one or more GPU kernels.  
* .rodata: This section contains read-only data used by the kernels. Critically, this is where the Kernel Descriptor for each kernel is stored.  
* Note Sections (.note): The ELF note mechanism is used to store structured metadata about the code object. This includes information about the version of the code object format and, most importantly, the target ISA for which the code was compiled. This is stored in an .hsa\_code\_object\_isa note, which specifies the major, minor, and stepping version of the GFX architecture (e.g., 9, 0, 6 for gfx906).

This standardized ELF format allows tools like readelf to inspect the contents of a GPU executable, and it provides a stable format for the HSA runtime's loader to parse and prepare for execution.

### **4.4. The GFX9 Kernel Descriptor: The Contract with Hardware**

Before the Command Processor can launch a kernel, it needs a detailed description of that kernel's properties and resource requirements. This information is provided in a 64-byte data structure called the Kernel Descriptor. This descriptor is generated by the compiler and stored in the .rodata section of the code object. It is arguably the most critical piece of metadata associated with a kernel, as it forms a direct contract between the compiled software and the hardware.10 An incorrect value in any field can lead to a failed launch, incorrect execution, or a hardware hang.

The LLVM AMDGPU Usage documentation provides a complete bit-level layout of this structure for GFX9.10 A programmer writing a custom assembler or code generation tool must be able to construct this structure perfectly. The key fields include:

* **KERNEL\_CODE\_ENTRY\_BYTE\_OFFSET:** A 64-bit value representing the byte offset from the start of the kernel descriptor itself to the first instruction of the kernel's machine code in the .text section. This must be 256-byte aligned.  
* **Resource Allocation (COMPUTE\_PGM\_RSRC1):** This 32-bit field contains several packed sub-fields that define the kernel's primary resource needs:  
  * GRANULATED\_WORKITEM\_VGPR\_COUNT: The number of VGPRs used by each thread. The hardware allocates VGPRs in blocks of 4\.  
  * GRANULATED\_WAVEFRONT\_SGPR\_COUNT: The number of SGPRs used by the wavefront. The hardware allocates SGPRs in blocks of 16\.  
  * These two values are critical for performance, as they determine the "occupancy"—how many wavefronts can be resident on a CU simultaneously.  
* **Hardware Setup (COMPUTE\_PGM\_RSRC2):** This 32-bit field contains a series of bit-flags that instruct the hardware on how to set up the initial state for the wavefronts:  
  * ENABLE\_SGPR\_WORKGROUP\_ID\_X/Y/Z: If set, the hardware will pre-load SGPRs with the work-group's ID.  
  * ENABLE\_VGPR\_WORKITEM\_ID: A 2-bit field that tells the hardware to pre-load VGPRs with the thread's local ID within the work-group.  
  * USER\_SGPR\_COUNT: The number of user SGPRs that will be pre-loaded with kernel arguments.  
* **Memory Requirements:**  
  * GROUP\_SEGMENT\_FIXED\_SIZE: The amount of LDS memory (in bytes) that must be allocated for each work-group.  
  * PRIVATE\_SEGMENT\_FIXED\_SIZE: The amount of scratch memory (in bytes) required per thread for register spills.  
* **Extended Enable Flags:** A series of single-bit flags located after the main resource words, such as ENABLE\_SGPR\_KERNARG\_SEGMENT\_PTR, which enables the pre-loading of the pointer to the kernel argument buffer.

The kernel descriptor is the essential bridge between the static, compiled code object and the dynamic, executing hardware. Its precise and correct construction is a non-negotiable requirement for low-level programming.

| Byte Offset | Bit Range | Field Name | Description |
| :---- | :---- | :---- | :---- |
| 0-3 | 31:0 | GROUP\_SEGMENT\_FIXED\_SIZE | Fixed Local Data Share (LDS) memory required for a work-group, in bytes. |
| 4-7 | 63:32 | PRIVATE\_SEGMENT\_FIXED\_SIZE | Fixed private (scratch) memory required for a single work-item, in bytes. |
| 8-11 | 95:64 | KERNARG\_SIZE | Size of the kernel argument memory region, in bytes. |
| 16-23 | 191:128 | KERNEL\_CODE\_ENTRY\_BYTE\_OFFSET | 64-bit byte offset from the descriptor's base to the kernel's entry point. Must be 256-byte aligned. |
| 48-51 | 415:384 | COMPUTE\_PGM\_RSRC1 | Packed 32-bit field for primary resource settings, including VGPR and SGPR counts, and floating-point modes. |
| 52-55 | 447:416 | COMPUTE\_PGM\_RSRC2 | Packed 32-bit field for hardware setup flags, including enabling system SGPRs/VGPRs and exception handling. |
| 56 | 448 | ENABLE\_SGPR\_PRIVATE\_SEGMENT\_BUFFER | Enables setup of the SGPR pointing to the private segment buffer. |
| 56 | 449 | ENABLE\_SGPR\_DISPATCH\_PTR | Enables setup of the SGPR pointing to the dispatch packet. |
| 56 | 450 | ENABLE\_SGPR\_QUEUE\_PTR | Enables setup of the SGPR pointing to the AQL queue. |
| 56 | 451 | ENABLE\_SGPR\_KERNARG\_SEGMENT\_PTR | Enables setup of the SGPR pointing to the kernel argument buffer. |
| 56 | 452 | ENABLE\_SGPR\_DISPATCH\_ID | Enables setup of the SGPR containing the dispatch ID. |
| 56 | 453 | ENABLE\_SGPR\_FLAT\_SCRATCH\_INIT | Enables setup of the SGPR for flat scratch initialization. |
| 56 | 454 | ENABLE\_SGPR\_PRIVATE\_SEGMENT\_SIZE | Enables setup of the SGPR containing the private segment size. |
| 57 | 459 | USES\_DYNAMIC\_STACK | Indicates if the kernel uses a dynamically sized stack. |

## **Section 5: Command Submission via the Architected Queuing Language (AQL)**

With a compiled and packaged kernel ready for execution, the final step is to instruct the GPU to run it. In the Heterogeneous System Architecture (HSA), this is achieved through a low-latency, user-mode command submission mechanism. The language used to communicate with the GPU's command processor is the Architected Queuing Language (AQL). Understanding the structure of AQL packets and the mechanics of the submission process is the key to unlocking direct, low-level control of the hardware.

### **5.1. User-Mode Queues and the Command Processor**

A central design philosophy of HSA is to minimize the overhead of dispatching work to the GPU. In older graphics APIs, every command submission often required a transition into the operating system kernel (a system call), which introduced significant latency. HSA eliminates this bottleneck by implementing user-mode queues.29

The process begins when an application uses the HSA runtime API (e.g., hsa\_queue\_create) to request a command queue from the driver. The amdgpu kernel driver, in response, allocates a region of memory (typically in system RAM) for the queue and maps it into both the application's virtual address space and the GPU's virtual address space. This shared memory region is structured as a ring buffer, which will hold the AQL packets.34 The driver also provides the application with a memory-mapped "doorbell" address.

From this point on, the submission process occurs entirely in user space. The application, acting as the "producer," writes one or more 64-byte AQL packets directly into the ring buffer. To do this, it first atomically increments the queue's write\_index to reserve space, then writes the packet data. Once the packet is written, the application "rings the doorbell" by writing the new write\_index to the special doorbell address.33 This doorbell write is the only action that directly signals the hardware. The GPU's Command Processor, acting as the "consumer," monitors this doorbell. When it detects a write, it knows that new packets are available in the queue up to the specified

write\_index, and it begins fetching and processing them. This entire sequence—reserving a slot, writing a packet, and ringing the doorbell—avoids any kernel-mode transitions, enabling extremely low-latency dispatch.

### **5.2. AQL Packet Structure: The Language of the GPU**

The AQL packet format is architected by the HSA Foundation, meaning it is a stable, cross-vendor standard. The full specification is detailed in the HSA Platform System Architecture Specification.36 All packets are 64 bytes in size.

**The Common Packet Header (Bytes 0-1):** The first 16 bits of every AQL packet form a common header that contains essential control information.

* format (8 bits): An enumeration that identifies the type of the packet. Key formats include KERNEL\_DISPATCH, BARRIER\_AND, BARRIER\_OR, and VENDOR\_SPECIFIC.  
* barrier (1 bit): A simple but powerful flag. If set, the Command Processor will not begin processing this packet until all preceding packets in the queue have fully completed. This enforces a strict in-order execution barrier.  
* acquire\_fence\_scope and release\_fence\_scope (2 bits each): These fields control the memory fence semantics associated with the packet. An acquire fence ensures that memory writes from other agents become visible before the packet's payload executes. A release fence ensures that memory writes from this packet's payload become visible to other agents after it completes. The scope (agent or system) determines the extent of this visibility.

**The Kernel Dispatch Packet (HSA\_PACKET\_TYPE\_KERNEL\_DISPATCH):** This is the most common and important packet type. It contains all the information the Command Processor needs to launch a computational kernel.

* dimensions (2 bits): The number of dimensions in the compute grid (1, 2, or 3).  
* workgroup\_size\_x/y/z (16 bits each): The size of each work-group in threads.  
* grid\_size\_x/y/z (32 bits each): The total size of the grid in threads.  
* private\_segment\_size\_bytes (32 bits): The amount of scratch memory required per thread. This must match the value in the kernel's descriptor.  
* group\_segment\_size\_bytes (32 bits): The amount of LDS required per work-group. This must also match the kernel descriptor.  
* kernel\_object (64 bits): This is an opaque handle that is effectively a pointer to the loaded kernel code object in memory.  
* kernarg\_address (64 bits): A pointer to the memory region where the kernel's arguments have been placed by the host application.  
* completion\_signal (64 bits): An optional handle to an HSA signal object. If non-zero, the hardware will atomically decrement the value of this signal object once the entire kernel dispatch has completed. This is the primary mechanism for the host to be notified of kernel completion.

**Barrier AND/OR Packets:** These packets provide a more flexible mechanism for synchronization than the simple barrier bit. They are used to create complex dependency graphs between kernels, potentially from different queues.

* Each barrier packet contains five 64-bit dep\_signal fields. Each field can hold the handle of an HSA signal object.  
* A **Barrier-AND** packet will stall the queue until *all* of its non-null dependency signals have been satisfied (typically by being decremented to zero by a completed kernel).  
* A Barrier-OR packet will stall the queue until any one of its non-null dependency signals has been satisfied.  
  These barrier packets enable the construction of Directed Acyclic Graphs (DAGs) of computation that can be submitted to the hardware and executed with minimal host intervention.

The existence of a formal, architected language like AQL is a cornerstone of low-level programming on AMD GPUs. High-level runtimes like HIP and OpenCL are, in essence, sophisticated AQL packet generators.34 Their launch API calls (

hipLaunchKernel, etc.) are ultimately translated into the construction and submission of a kernel\_dispatch\_packet. By learning to construct these packets manually, a programmer can bypass the runtime abstractions entirely and communicate with the hardware at the same fundamental level. This provides the ultimate degree of control over dispatch, synchronization, and memory fencing, allowing for the implementation of custom schedulers, the elimination of runtime overhead, and the fine-grained orchestration of complex, multi-kernel workflows. This is the practical endpoint of the desire to "program at a low level."

| Byte Offset | Bit Range | Field Name | Description |
| :---- | :---- | :---- | :---- |
| 0-1 | 15:0 | header | Packet header, containing format (2 for kernel dispatch), barrier bit, and acquire/release fence scopes. |
| 2-3 | 17:16 | dimensions | Number of dimensions in the grid (1, 2, or 3). |
| 4-5 | 47:32 | workgroup\_size\_x | X-dimension of the work-group size in threads. |
| 6-7 | 63:48 | workgroup\_size\_y | Y-dimension of the work-group size in threads. |
| 8-9 | 79:64 | workgroup\_size\_z | Z-dimension of the work-group size in threads. |
| 12-15 | 127:96 | grid\_size\_x | X-dimension of the grid size in threads. |
| 16-19 | 159:128 | grid\_size\_y | Y-dimension of the grid size in threads. |
| 20-23 | 191:160 | grid\_size\_z | Z-dimension of the grid size in threads. |
| 24-27 | 223:192 | private\_segment\_size\_bytes | Bytes of private (scratch) memory required per work-item. |
| 28-31 | 255:224 | group\_segment\_size\_bytes | Bytes of group (LDS) memory required per work-group. |
| 32-39 | 319:256 | kernel\_object | 64-bit opaque handle (pointer) to the loaded kernel code object. |
| 40-47 | 383:320 | kernarg\_address | 64-bit pointer to the memory buffer containing kernel arguments. |
| 56-63 | 511:448 | completion\_signal | 64-bit opaque handle to an HSA signal object for completion notification. |

## **Section 6: The Foundation: The amdgpu Linux Kernel Driver**

At the absolute lowest level of the software stack sits the kernel-mode driver (KMD). For modern AMD GPUs on Linux, this is the amdgpu driver, which is part of the mainline Linux kernel. While a low-level application programmer typically interacts with the user-space HSA runtime rather than the KMD directly, an understanding of the driver's role and structure is essential for deep system analysis, debugging, and for appreciating the full hardware-software contract. The driver's source code also serves as the ultimate, albeit complex, source of hardware documentation.

### **6.1. Role and Responsibilities of the KMD**

The amdgpu driver is a privileged component of the operating system that has exclusive, direct access to the GPU's hardware registers and command submission mechanisms.38 Its primary responsibilities include 39:

* **Device Initialization and Firmware Loading:** When the system boots or the driver is loaded, amdgpu probes the PCIe bus for supported devices. Upon finding a GPU, it initiates a complex initialization sequence. This includes loading various firmware blobs required by the GPU's onboard microcontrollers, such as the Platform Security Processor (PSP), the System Management Unit (SMU), and the Graphics and Compute Microcontrollers.39 It then initializes the core IP blocks of the GPU, such as the graphics (GFX) engine, the memory hub (MMHUB), and the display controllers.  
* **Memory Management:** The driver is the sole manager of the GPU's physical memory resources. It manages the allocation of Video RAM (VRAM) and the Graphics Address Remapping Table (GART), which is a portion of system RAM made accessible to the GPU.41 It implements the GPU Virtual Memory (GPUVM) system, creating and managing the page tables that translate virtual addresses used by applications into physical addresses in VRAM or GART.  
* **Queue and Context Management:** The driver is responsible for creating the hardware contexts and queues that the GPU's command processors use. When a user-space application requests an AQL queue via the HSA runtime, the amdgpu driver allocates the necessary hardware resources and maps the queue's ring buffer and doorbell into the application's address space. It is responsible for scheduling and multiplexing the potentially numerous software queues from multiple processes onto the limited number of physical hardware queues.35  
* **Interrupt Handling and Error Recovery:** The driver sets up and services interrupts from the GPU. These interrupts signal important events such as the completion of a command buffer, a page fault in GPUVM, or a hardware error. In the event of a GPU hang, the driver is responsible for attempting to reset the GPU and recover the system to a stable state.  
* **Power Management:** The driver communicates with the SMU to manage the GPU's power states, clock frequencies, and fan speeds. It exposes interfaces through sysfs that allow user-space tools to monitor and, to some extent, control these parameters.39

### **6.2. Navigating the Driver Source: A Programmer's Map**

For the determined low-level programmer or reverse engineer, the amdgpu driver source code is the most comprehensive technical reference available. The source is located within the Linux kernel tree at drivers/gpu/drm/amd/amdgpu/.42 Navigating this large and complex codebase requires a map of the key files relevant to the GFX9 architecture.

* **Core GFX9 Implementation:**  
  * gfx\_v9\_0.c: This file contains the GFX-specific implementation for the Vega 10 family of GPUs, which forms the basis for Vega 20 (gfx906). It includes functions for initializing the GFX hardware block, managing the graphics and compute ring buffers, parsing command buffers, and handling GFX-related interrupts.43  
* **SoC-Level Implementation:**  
  * soc15.c: The Vega architecture is part of the "SOC15" family of AMD ASICs. This file contains common functions and data structures that are shared across all SOC15-based GPUs, including Vega (GFX9) and Navi (GFX10). It handles initialization of IP blocks that are common to the SoC, such as the memory hub.45  
* **Driver Infrastructure:**  
  * amdgpu\_device.c: This file contains the high-level logic for device discovery, initialization, and teardown.47  
  * amdgpu\_ring.c: Implements the generic logic for managing command ring buffers, which are used by all hardware engines (GFX, compute, SDMA).  
  * amdgpu\_vm.c: Contains the implementation of the GPU Virtual Memory manager.

A notable characteristic of the amdgpu driver is its immense size, a significant portion of which is composed of auto-generated C header files.1 These headers, often named after the IP blocks they describe (e.g.,

gfx\_9\_0\_sh\_mask.h), contain thousands of \#define macros. These macros define the memory-mapped register offsets for every controllable aspect of the hardware, as well as the bit-field masks and shifts for individual settings within those registers.

While this "documentation as code" approach makes the driver source tree unwieldy, it provides an unparalleled resource. The kernel headers represent the most complete and accurate public documentation of the GFX9 hardware register map. For a programmer seeking to understand a specific hardware behavior or to interact with a register not exposed by any higher-level API, searching through these headers within the kernel source is often the only way to find the necessary register addresses and bit-field definitions. They are the ultimate ground truth for hardware control.

### **6.3. Driver Data Structures: Rings and IBs**

It is important to distinguish between the user-mode AQL queues used by the HSA runtime and the kernel-mode ring buffers managed directly by the amdgpu driver. The driver maintains its own set of ring buffers for each hardware engine (e.g., a gfx ring, multiple compute rings, sdma rings for DMA transfers).38

The driver writes commands to these rings to perform privileged operations that a user-space application cannot, such as setting up page tables or triggering a context switch. These kernel-level commands are written in a format called PM4. When a user-space application submits work (e.g., via an AQL queue or a Vulkan command buffer), the submission is typically packaged into an Indirect Buffer (IB).38 The driver then validates this IB and writes a small PM4 packet to its own ring buffer. This packet, often an

INDIRECT\_BUFFER command, simply contains a pointer to the user-space IB and its size. This tells the GPU's command processor to switch context, jump to the address of the IB, and begin executing the user-provided commands.38 This two-level system maintains a security boundary while still allowing for efficient submission of large command buffers from user space.

### **6.4. The sysfs Interface: Monitoring and Control**

The amdgpu driver exposes a wealth of information and control knobs through the Linux sysfs pseudo-filesystem, typically located under /sys/class/drm/cardX/device/ (where X is the card number).39 This provides a standardized, file-based interface for monitoring and tweaking the GPU's state.

Key sysfs interfaces for a low-level programmer include:

* **Memory Information:**  
  * mem\_info\_vram\_total, mem\_info\_vram\_used: Report the total and used VRAM in bytes.  
  * mem\_info\_gtt\_total, mem\_info\_gtt\_used: Report the total and used GART/GTT memory in bytes.41  
* **Power Management:**  
  * power\_dpm\_force\_performance\_level: Allows a user with sufficient privileges to lock the GPU's performance level to a specific state (e.g., 'high', 'low', 'auto'), which can be useful for achieving deterministic performance during benchmarking.  
  * pp\_od\_clk\_voltage: Exposes an interface for overclocking by allowing manual adjustment of frequency/voltage points.  
  * gpu\_metrics: A comprehensive file that provides a detailed snapshot of the GPU's current state, including temperatures, clock speeds for various domains (GPU core, memory), fan speed, and power consumption.  
* **Device Identification:**  
  * unique\_id: For GFX9 and newer GPUs, this file provides a persistent, unique identifier for the specific GPU device, which can be useful for identifying a particular card in a multi-GPU system.51

These sysfs interfaces are invaluable for debugging and performance analysis, providing a direct window into the hardware's real-time operational state as managed by the kernel driver.

## **Section 7: Recommendations and Practical Strategy**

Having explored the GFX906 architecture from the silicon up to the kernel driver, this final section synthesizes these technical details into a pragmatic and actionable strategy for the low-level programmer. The path to direct hardware control is challenging, particularly for a device like the Instinct MI50, which has passed its official support window. Success requires a phased approach, a specific set of tools, and a clear understanding of the practical limitations.

### **7.1. A Phased Approach to Low-Level Programming**

A direct leap into writing raw AQL packets is likely to be unproductive. A more structured, incremental approach is recommended to build the necessary foundation and toolchain.

Phase 1: Establish a Functional Baseline  
The first and most critical step is to create a stable, working environment. This involves addressing both physical and software prerequisites.

1. **Hardware Setup:** The Instinct MI50 is a server-grade accelerator and has specific hardware requirements. It is a passively cooled card that requires a high-airflow server chassis. It may not POST (Power-On Self-Test) in many consumer-grade motherboards due to firmware incompatibilities.52 Success often requires a compatible server motherboard with appropriate BIOS settings (e.g., enabling Above 4G Decoding). In some cases, users have resorted to cross-flashing the card's firmware to that of a Radeon Pro VII to improve compatibility, though this is a high-risk procedure that can permanently damage the card.53  
2. **Software Installation:** The gfx906 architecture entered "maintenance mode" with the ROCm 5.7 release in Q3 2023 and reached its "End of Maintenance" (EOM) in Q2 2024\.8 This means that the latest versions of the ROCm stack do not officially support this hardware. The programmer must install a version of ROCm known to be compatible, such as ROCm 5.7 or an earlier release.  
3. **Verification:** Once the hardware is physically installed and the software is set up, use the standard ROCm utilities to verify that the system is functional. Running rocminfo should list the gfx906 agent, and rocm-smi should report the card's status, temperature, and memory usage.55 Establishing this baseline is crucial before proceeding to more advanced programming.

Phase 2: Analysis and Exploration via High-Level APIs  
Before writing low-level code, it is immensely valuable to study the output of the existing toolchain.

1. **Write Simple Kernels:** Author simple compute kernels using HIP or OpenCL. These high-level models handle the complexities of compilation, packaging, and dispatch.  
2. **Dump the Artifacts:** Use the ROCm compiler's flags (e.g., clang \--offload-arch=gfx906 \-save-temps) to instruct it to save the intermediate files generated during compilation. This will produce the GCN assembly (.s file) and the final HSA code object (.o file).  
3. **Study the Output:** Carefully analyze the generated assembly to understand how high-level constructs are translated into the GFX9 ISA. Use tools like readelf to inspect the structure of the HSA code object, paying close attention to the kernel descriptor in the .rodata section. This phase provides a set of known-good examples of what correct, low-level code and metadata look like.

Phase 3: Inline GCN Assembly  
The next step is to begin writing ISA code directly, but within the managed environment of a higher-level language.

1. **Use Inline asm:** The HIP C++ language supports inline assembly statements, similar to standard C++. This allows the programmer to write small snippets of GCN assembly directly within a \_\_global\_\_ kernel function.  
2. **Experiment with Instructions:** This is the ideal environment to experiment with specific instructions, test operand combinations, and understand the behavior of scalar and vector operations without having to build an entire kernel from scratch. The ROCm compiler and runtime still handle the boilerplate of creating the kernel descriptor and dispatching the kernel.

Phase 4: Manual Command Submission via HSA Runtime  
This final phase achieves the ultimate goal of direct, low-level control.

1. **Use the HSA API:** Write a host program in C or C++ that links directly against the HSA runtime library (libhsa-runtime64.so).  
2. **Manual Orchestration:** The program will use the HSA API to perform the full dispatch sequence manually: initialize the runtime, discover the gfx906 agent, create an AQL queue, allocate GPU-visible memory (for arguments and output), load a pre-compiled HSA code object, and get a handle to the kernel\_object.  
3. **Construct and Submit AQL Packets:** The core of the program will be a loop that reserves a slot in the AQL queue's ring buffer, manually constructs a 64-byte hsa\_kernel\_dispatch\_packet\_t in that memory slot (as detailed in Section 5), and then rings the queue's doorbell to launch the kernel.  
4. **Synchronization:** Use HSA signal objects and the hsa\_signal\_wait\_acquire API call to wait for kernel completion.

Successfully completing this phase demonstrates a mastery of the hardware's command submission interface, bypassing all high-level abstractions and interacting with the GPU at the same level as the ROCm runtime itself.

### **7.2. Essential Toolchain and Resources**

A successful low-level programming effort for the MI50 requires a specific set of software tools and documentation.

**Software Toolkit:**

* A supported Linux distribution (e.g., Ubuntu 20.04/22.04, RHEL 8/9) compatible with the chosen ROCm version.55  
* ROCm version 5.7 or an earlier, compatible release.  
* The LLVM/Clang toolchain, which is included with ROCm, for its amdgcn backend.  
* A local clone of the Linux kernel source repository, for browsing the amdgpu driver source and its invaluable register definition headers.  
* Standard binary analysis tools like readelf and a hex editor for inspecting code objects and memory.

**Documentation Library:**

* **Primary (Essential for Implementation):**  
  1. **HSA Platform System Architecture Specification:** The definitive source for the AQL packet format and user-mode queuing mechanics.36  
  2. **LLVM AMDGPU Backend Documentation & Source:** The ground truth for ISA syntax, operand formats, and the GFX9 memory model.7 The source code itself (  
     .td files) is the only reference for hardware errata.18  
  3. **amdgpu Kernel Driver Source Code:** The ultimate reference for hardware register maps and initialization sequences.  
* **Secondary (Essential for Concepts):**  
  1. **AMD "Vega" ISA PDF:** Provides the high-level architectural context and conceptual understanding of the instruction set.6  
  2. **AMD "Vega" Architecture Whitepaper:** Explains the design philosophy and key features like the HBCC and Infinity Fabric.5

### **7.3. Caveats and Advanced Topics: The Uncharted Territory**

Finally, it is crucial to acknowledge the significant challenges and limitations inherent in this endeavor.

**End-of-Maintenance Status:** The most significant caveat is the gfx906 architecture's EOM status.8 There will be no new official features, performance optimizations, or bug fixes from AMD. The programmer is reliant on the existing software, community support, and their own ability to debug issues.

**Firmware and the Platform Security Processor (PSP):** Modern GPUs are not monolithic processors; they contain multiple microcontrollers that run their own firmware. The PSP is a dedicated ARM processor responsible for secure boot, firmware loading, and other security-critical tasks.57 The VBIOS and other firmware components are cryptographically signed. This makes any attempt to modify the firmware (e.g., to change the device ID or unlock features) extremely difficult, as it would require breaking this chain of trust. Without a hardware-level exploit, VBIOS modification on Vega is generally considered infeasible.59

**The Pragmatic Path:** The user's goal is to "program at a low level." This could be interpreted as a desire to write a custom kernel driver from scratch. However, given the immense complexity of the amdgpu driver, which spans millions of lines of code handling everything from power management to memory virtualization, this is not a practical undertaking.39 The most effective and pragmatic path to low-level control is to leverage the existing, open-source

amdgpu driver and ROCm/HSA stack. The HSA standard was explicitly designed to provide a stable, low-latency, user-space interface for command submission. By targeting the HSA runtime API directly, a programmer can achieve direct control over the hardware's command processor—constructing and submitting their own AQL packets—without the insurmountable burden of developing and maintaining a custom kernel-mode driver. This approach represents the optimal balance of control, performance, and feasibility, and is the recommended path for any low-level programming on the Instinct MI50.

#### **Works cited**

1. Updated Vega 20 Open-Source Driver Patches Posted, Including PSP & PowerPlay Support, accessed August 14, 2025, [https://www.phoronix.com/news/Vega-20-More-Driver-Code](https://www.phoronix.com/news/Vega-20-More-Driver-Code)  
2. VEGA20 Linux patches : r/Amd \- Reddit, accessed August 14, 2025, [https://www.reddit.com/r/Amd/comments/88rmnz/vega20\_linux\_patches/](https://www.reddit.com/r/Amd/comments/88rmnz/vega20_linux_patches/)  
3. Graphics Core Next \- Wikipedia, accessed August 14, 2025, [https://en.wikipedia.org/wiki/Graphics\_Core\_Next](https://en.wikipedia.org/wiki/Graphics_Core_Next)  
4. AMD GPU Hardware Basics, accessed August 14, 2025, [https://www.olcf.ornl.gov/wp-content/uploads/2019/10/ORNL\_Application\_Readiness\_Workshop-AMD\_GPU\_Basics.pdf](https://www.olcf.ornl.gov/wp-content/uploads/2019/10/ORNL_Application_Readiness_Workshop-AMD_GPU_Basics.pdf)  
5. Radeon's next-generation Vega architecture \- WikiChip, accessed August 14, 2025, [https://en.wikichip.org/w/images/a/a1/vega-whitepaper.pdf](https://en.wikichip.org/w/images/a/a1/vega-whitepaper.pdf)  
6. "Vega" Instruction Set Architecture | AMD, accessed August 14, 2025, [https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/vega-shader-instruction-set-architecture.pdf](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/vega-shader-instruction-set-architecture.pdf)  
7. Syntax of gfx906 Instructions — LLVM 22.0.0git documentation, accessed August 14, 2025, [https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX906.html](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX906.html)  
8. Support your GPUs for 8+ years, like Nvidia does, including gfx906 GPUs · ROCm ROCm · Discussion \#3893 \- GitHub, accessed August 14, 2025, [https://github.com/ROCm/ROCm/discussions/3893](https://github.com/ROCm/ROCm/discussions/3893)  
9. Support your GPUs for 8+ years, like Nvidia does, including gfx906 GPUs · Issue \#2308 · ROCm/ROCm \- GitHub, accessed August 14, 2025, [https://github.com/RadeonOpenCompute/ROCm/issues/2308](https://github.com/RadeonOpenCompute/ROCm/issues/2308)  
10. User Guide for AMDGPU Backend — LLVM 22.0.0git documentation, accessed August 14, 2025, [https://llvm.org/docs/AMDGPUUsage.html](https://llvm.org/docs/AMDGPUUsage.html)  
11. AMD “Vega” 7nm Instruction Set Architecture documentation \- AMD ..., accessed August 14, 2025, [https://gpuopen.com/news/amd-vega-7nm-instruction-set-architecture-documentation/](https://gpuopen.com/news/amd-vega-7nm-instruction-set-architecture-documentation/)  
12. Syntax of Core GFX9 Instructions — LLVM 19.0.0git documentation, accessed August 14, 2025, [https://rocm.docs.amd.com/projects/llvm-project/en/develop/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX9.html](https://rocm.docs.amd.com/projects/llvm-project/en/develop/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX9.html)  
13. User Guide for AMDGPU Backend — LLVM 8 documentation, accessed August 14, 2025, [https://prereleases.llvm.org/8.0.0/rc3/docs/AMDGPUUsage.html](https://prereleases.llvm.org/8.0.0/rc3/docs/AMDGPUUsage.html)  
14. User Guide for AMDGPU Backend — LLVM 19.0.0git documentation, accessed August 14, 2025, [https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html)  
15. Syntax of Core GFX9 Instructions — LLVM 22.0.0git documentation, accessed August 14, 2025, [https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX9.html](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX9.html)  
16. Radeon "GFX9" Support Lands In LLVM's AMDGPU Backend \- Phoronix, accessed August 14, 2025, [https://www.phoronix.com/news/AMDGPU-LLVM-GFX9](https://www.phoronix.com/news/AMDGPU-LLVM-GFX9)  
17. Building AMD ROCm from Source on a Supercomputer \- Cray User Group, accessed August 14, 2025, [https://cug.org/proceedings/cug2023\_proceedings/includes/files/pap104s2-file1.pdf](https://cug.org/proceedings/cug2023_proceedings/includes/files/pap104s2-file1.pdf)  
18. llvm-project/llvm/lib/Target/AMDGPU/AMDGPU.td at main \- GitHub, accessed August 14, 2025, [https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPU.td](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPU.td)  
19. AMD machine-readable GPU ISA documentation, accessed August 14, 2025, [https://gpuopen.com/machine-readable-isa/](https://gpuopen.com/machine-readable-isa/)  
20. AMD GPU architecture programming documentation, accessed August 14, 2025, [https://gpuopen.com/amd-gpu-architecture-programming-documentation/](https://gpuopen.com/amd-gpu-architecture-programming-documentation/)  
21. Syntax of AMDGPU Instruction Operands — LLVM 19.0.0git documentation, accessed August 14, 2025, [https://rocm.docs.amd.com/projects/llvm-project/en/develop/LLVM/llvm/html/AMDGPUOperandSyntax.html](https://rocm.docs.amd.com/projects/llvm-project/en/develop/LLVM/llvm/html/AMDGPUOperandSyntax.html)  
22. gcn3-instruction-set-architecture.pdf \- AMD, accessed August 14, 2025, [https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/gcn3-instruction-set-architecture.pdf](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/gcn3-instruction-set-architecture.pdf)  
23. User Guide for AMDGPU Backend \- Read the Docs — bcain-llvm latest documentation, accessed August 14, 2025, [https://bcain-llvm.readthedocs.io/projects/llvm/en/latest/AMDGPUUsage/](https://bcain-llvm.readthedocs.io/projects/llvm/en/latest/AMDGPUUsage/)  
24. User Guide for AMDGPU Backend — LLVM 8 documentation, accessed August 14, 2025, [https://prereleases.llvm.org/8.0.0/rc5/docs/AMDGPUUsage.html](https://prereleases.llvm.org/8.0.0/rc5/docs/AMDGPUUsage.html)  
25. AMD ROCm™ Software, accessed August 14, 2025, [https://www.amd.com/en/products/software/rocm.html](https://www.amd.com/en/products/software/rocm.html)  
26. Programming guide — ROCm Documentation, accessed August 14, 2025, [https://rocm.docs.amd.com/en/latest/how-to/programming\_guide.html](https://rocm.docs.amd.com/en/latest/how-to/programming_guide.html)  
27. OpenCL Programming Guide — ROCm 4.5.0 documentation, accessed August 14, 2025, [https://cgmb-rocm-docs.readthedocs.io/en/latest/Programming\_Guides/Opencl-programming-guide.html](https://cgmb-rocm-docs.readthedocs.io/en/latest/Programming_Guides/Opencl-programming-guide.html)  
28. AMD ROCm / HCC programming: Introduction \- Reddit, accessed August 14, 2025, [https://www.reddit.com/r/Amd/comments/a9tjge/amd\_rocm\_hcc\_programming\_introduction/](https://www.reddit.com/r/Amd/comments/a9tjge/amd_rocm_hcc_programming_introduction/)  
29. ReadTheDocs-Breathe Documentation \- Read the Docs, accessed August 14, 2025, [https://readthedocs.org/projects/blas-testing/downloads/pdf/latest/](https://readthedocs.org/projects/blas-testing/downloads/pdf/latest/)  
30. HSA Runtime API and runtime for ROCm — ROCR 1.13.0 Documentation, accessed August 14, 2025, [https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-6.1.1/](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-6.1.1/)  
31. ROCR-Runtime/README.md at amd-staging \- GitHub, accessed August 14, 2025, [https://github.com/ROCm/ROCR-Runtime/blob/amd-staging/README.md](https://github.com/ROCm/ROCR-Runtime/blob/amd-staging/README.md)  
32. AMDGPU LLVM Adding GFX 9/10/11 "Generic Targets" To Build Once & Run On Multiple GPUs \- Phoronix, accessed August 14, 2025, [https://www.phoronix.com/news/LLVM-AMDGPU-Generic-GFX](https://www.phoronix.com/news/LLVM-AMDGPU-Generic-GFX)  
33. hsa queueing \- Hot Chips, accessed August 14, 2025, [https://old.hotchips.org/wp-content/uploads/hc\_archives/hc25/HC25.0T1-Hetero-epub/HC25.25.130-Queuing-bratt-HSA%20Queuing%20HotChips2013\_Final.pdf](https://old.hotchips.org/wp-content/uploads/hc_archives/hc25/HC25.0T1-Hetero-epub/HC25.25.130-Queuing-bratt-HSA%20Queuing%20HotChips2013_Final.pdf)  
34. Exploring AMD GPU Scheduling Details by Experimenting With “Worst Practices”, accessed August 14, 2025, [https://par.nsf.gov/servlets/purl/10385873](https://par.nsf.gov/servlets/purl/10385873)  
35. Documentation about AMD's HSA implementation? \- Mailing Lists \- Freedesktop.org, accessed August 14, 2025, [https://lists.freedesktop.org/archives/amd-gfx/2018-February/019035.html](https://lists.freedesktop.org/archives/amd-gfx/2018-February/019035.html)  
36. HSA Platform System Architecture Specification ... \- HSA Foundation, accessed August 14, 2025, [http://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf](http://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf)  
37. AMD Debugger API \- ROCm Documentation, accessed August 14, 2025, [https://rocm.docs.amd.com/projects/ROCdbgapi/en/latest/doxygen/html/index.html](https://rocm.docs.amd.com/projects/ROCdbgapi/en/latest/doxygen/html/index.html)  
38. RADV — The Mesa 3D Graphics Library latest documentation, accessed August 14, 2025, [https://docs.mesa3d.org/drivers/radv.html](https://docs.mesa3d.org/drivers/radv.html)  
39. drm/amdgpu AMDgpu driver \- The Linux Kernel documentation, accessed August 14, 2025, [https://docs.kernel.org/gpu/amdgpu/index.html](https://docs.kernel.org/gpu/amdgpu/index.html)  
40. drm/amdgpu AMDgpu driver — The Linux Kernel documentation, accessed August 14, 2025, [https://dri.freedesktop.org/docs/drm/gpu/amdgpu/index.html](https://dri.freedesktop.org/docs/drm/gpu/amdgpu/index.html)  
41. drm/amdgpu AMDgpu driver — The Linux Kernel documentation, accessed August 14, 2025, [https://www.kernel.org/doc/html/v5.9/gpu/amdgpu.html](https://www.kernel.org/doc/html/v5.9/gpu/amdgpu.html)  
42. amdgpu\_drv.c source code \[linux/drivers/gpu/drm/amd/amdgpu ..., accessed August 14, 2025, [https://codebrowser.dev/linux/linux/drivers/gpu/drm/amd/amdgpu/amdgpu\_drv.c.html](https://codebrowser.dev/linux/linux/drivers/gpu/drm/amd/amdgpu/amdgpu_drv.c.html)  
43. PSA: Avoid Kernel 5.12.13/5.10.46/5.13-rc7 If Using AMD GFX9/GFX10 (Vega, Navi) GPUs : r/archlinux \- Reddit, accessed August 14, 2025, [https://www.reddit.com/r/archlinux/comments/o7x5j8/psa\_avoid\_kernel\_5121351046513rc7\_if\_using\_amd/](https://www.reddit.com/r/archlinux/comments/o7x5j8/psa_avoid_kernel_5121351046513rc7_if_using_amd/)  
44. accessed December 31, 1969, [https://elixir.bootlin.com/linux/latest/source/drivers/gpu/drm/amd/amdgpu/gfx\_v9\_0.c](https://elixir.bootlin.com/linux/latest/source/drivers/gpu/drm/amd/amdgpu/gfx_v9_0.c)  
45. Increasing VFIO VGA Performance \- \#176 by gnif \- Linux, accessed August 14, 2025, [https://forum.level1techs.com/t/increasing-vfio-vga-performance/133443/176](https://forum.level1techs.com/t/increasing-vfio-vga-performance/133443/176)  
46. \[Meta\] Support for Intel, Nouveau and radeon GPUs · Issue \#106 · Syllo/nvtop \- GitHub, accessed August 14, 2025, [https://github.com/Syllo/nvtop/issues/106](https://github.com/Syllo/nvtop/issues/106)  
47. ROCK-Kernel-Driver/drivers/gpu/drm/amd/amdgpu/amdgpu\_device.c at master \- GitHub, accessed August 14, 2025, [https://github.com/ROCm/ROCK-Kernel-Driver/blob/master/drivers/gpu/drm/amd/amdgpu/amdgpu\_device.c](https://github.com/ROCm/ROCK-Kernel-Driver/blob/master/drivers/gpu/drm/amd/amdgpu/amdgpu_device.c)  
48. Idea Raised For Reducing The Size Of The AMDGPU Driver With Its Massive Header Files, accessed August 14, 2025, [https://www.phoronix.com/news/AMDGPU-Headers-Repo-Idea](https://www.phoronix.com/news/AMDGPU-Headers-Repo-Idea)  
49. The AMD Radeon Graphics Driver Makes Up Roughly 10.5% Of The Linux Kernel \- Reddit, accessed August 14, 2025, [https://www.reddit.com/r/linux\_gaming/comments/j9hjqm/the\_amd\_radeon\_graphics\_driver\_makes\_up\_roughly/](https://www.reddit.com/r/linux_gaming/comments/j9hjqm/the_amd_radeon_graphics_driver_makes_up_roughly/)  
50. \[PATCH 2/4\] drm/amdgpu: Add software ring callbacks for gfx9 (v7) \- Mailing Lists, accessed August 14, 2025, [https://lists.freedesktop.org/archives/amd-gfx/2022-September/084846.html](https://lists.freedesktop.org/archives/amd-gfx/2022-September/084846.html)  
51. Misc AMDGPU driver information — The Linux Kernel documentation, accessed August 14, 2025, [https://dri.freedesktop.org/docs/drm/gpu/amdgpu/driver-misc.html](https://dri.freedesktop.org/docs/drm/gpu/amdgpu/driver-misc.html)  
52. Interesting cheap GPU option: Instinct Mi50 : r/LocalLLaMA \- Reddit, accessed August 14, 2025, [https://www.reddit.com/r/LocalLLaMA/comments/1b5ie1t/interesting\_cheap\_gpu\_option\_instinct\_mi50/](https://www.reddit.com/r/LocalLLaMA/comments/1b5ie1t/interesting_cheap_gpu_option_instinct_mi50/)  
53. Running local AI on AMD Instinct mi50 16gb, can it be done? \- GPU \- Level1Techs Forums, accessed August 14, 2025, [https://forum.level1techs.com/t/running-local-ai-on-amd-instinct-mi50-16gb-can-it-be-done/224892](https://forum.level1techs.com/t/running-local-ai-on-amd-instinct-mi50-16gb-can-it-be-done/224892)  
54. Help Flash MI50 to Radeon VII Pro | TechPowerUp Forums, accessed August 14, 2025, [https://www.techpowerup.com/forums/threads/help-flash-mi50-to-radeon-vii-pro.329623/](https://www.techpowerup.com/forums/threads/help-flash-mi50-to-radeon-vii-pro.329623/)  
55. Installation prerequisites — ROCm installation (Linux) \- ROCm Documentation \- AMD, accessed August 14, 2025, [https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html)  
56. Doesn't ROCm support AMD's integrated GPU (APU)? · Issue \#2216 \- GitHub, accessed August 14, 2025, [https://github.com/ROCm/ROCm/issues/2216](https://github.com/ROCm/ROCm/issues/2216)  
57. More Vega 20 Enablement Heading To Linux 4.20\~5.0, No Longer Marked Experimental, accessed August 14, 2025, [https://www.phoronix.com/news/More-Vega-20-Enablement-Linux](https://www.phoronix.com/news/More-Vega-20-Enablement-Linux)  
58. Reversing the AMD Secure Processor (PSP) \- Part 1: Design and Overview \- dayzerosec, accessed August 14, 2025, [https://dayzerosec.com/blog/2023/04/17/reversing-the-amd-secure-processor-psp.html](https://dayzerosec.com/blog/2023/04/17/reversing-the-amd-secure-processor-psp.html)  
59. GPU Firmware Hacking/Reverse Engineering Thread \- GPU ..., accessed August 14, 2025, [https://forum.level1techs.com/t/gpu-firmware-hacking-reverse-engineering-thread/134211](https://forum.level1techs.com/t/gpu-firmware-hacking-reverse-engineering-thread/134211)  
60. Reverse-Engineering The AMD Secure Processor Inside The CPU \- Hackaday, accessed August 14, 2025, [https://hackaday.com/2024/08/18/reverse-engineering-the-amd-secure-processor-inside-the-cpu/](https://hackaday.com/2024/08/18/reverse-engineering-the-amd-secure-processor-inside-the-cpu/)