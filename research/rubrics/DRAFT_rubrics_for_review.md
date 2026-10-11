# Concept rubrics for VBCUA — DRAFT – NOT FACULTY REVIEWED

> **DRAFT – NOT FACULTY REVIEWED.** These rubrics were drafted by the student researcher and have not yet been
> checked by a subject expert. They must not be used to grade anyone until reviewed.

## What these are for

VBCUA asks a student to explain a computer-science concept aloud for 1–3 minutes. The
recording is transcribed and the explanation is scored. Today the score is a sentence-
embedding similarity to a reference description. The research question (RQ1) is whether
a rubric — the key points a good explanation contains, plus common misconceptions —
agrees better with human markers than that similarity score.

## How a rubric is applied

- Each **key point** is marked **0** (missing or wrong), **1** (partly stated or vague) or
  **2** (clearly and correctly stated). The *partial credit* column says what earns a 1.
- Each **misconception** is marked as stated or not stated.
- Proposed overall score: score = max(0, sum of key-point marks − number of misconceptions stated) ÷ (2 × number of key points) × 100.
- Explanations are spoken, so wording, order and informal language should not matter; only
  whether the idea is there and correct.

## What I would like you to check

1. Are the key points correct, and are they the right ones for an undergraduate explanation?
   Should any be added, removed, merged or split?
2. Is a 1–3 minute spoken explanation expected to cover all of them, or should some be optional?
3. Are the misconceptions ones students really state? Are there more common ones?
4. Is the partial-credit description for each key point a fair boundary between 1 and 2?
5. Is the proposed overall score (with a 1-mark deduction per misconception) reasonable?

Please mark changes directly on this document or in comments.

## Deadlock

*DRAFT – NOT FACULTY REVIEWED* · file `research/rubrics/deadlock.json`

| # | Key point (2 marks if clearly and correctly stated) | Partial credit (1 mark) if the student… |
|---|---|---|
| K1 | A deadlock is a state in which a set of processes or threads each wait for a resource held by another member of the set, so none of them can ever proceed. | says processes are 'stuck' or 'waiting' without saying they wait on each other |
| K2 | All four (Coffman) conditions must hold at once: mutual exclusion, hold and wait, no preemption, circular wait. | names some of the conditions, or all four without saying all must hold together |
| K3 | A concrete example, such as two threads that each hold one lock and request the lock the other holds. | gives an example that is incomplete or not clearly a circular wait |
| K4 | Prevention: make one condition impossible, e.g. acquire locks in a fixed global order (no circular wait) or request all resources at once (no hold and wait). | says deadlocks can be prevented but gives no mechanism |
| K5 | Avoidance and detection: avoid unsafe states (e.g. Banker's algorithm), or detect cycles in a wait-for graph and recover by aborting or preempting a process. | mentions avoidance or detection by name only |

| # | Misconception (flag if stated) |
|---|---|
| M1 | Deadlock is the same as starvation, or simply a process that is slow or busy. |
| M2 | A deadlock resolves itself if the processes wait long enough. |
| M3 | Any single condition, such as mutual exclusion alone, is enough to cause a deadlock. |

Maximum: 10 marks.

## Database normalisation

*DRAFT – NOT FACULTY REVIEWED* · file `research/rubrics/normalisation.json`

| # | Key point (2 marks if clearly and correctly stated) | Partial credit (1 mark) if the student… |
|---|---|---|
| K1 | Normalisation organises the columns and tables of a relational database to reduce redundant (duplicated) data. | says it 'organises' or 'cleans' the data without mentioning redundancy or duplication |
| K2 | Redundancy causes update, insertion and deletion anomalies (e.g. changing a fact in one row but not another), and normalisation prevents them. | says data can become inconsistent but names no anomaly or example |
| K3 | Tables are decomposed into smaller related tables linked by primary and foreign keys, without losing information (the original data can be rebuilt with joins). | says tables are split but does not mention keys or that nothing is lost |
| K4 | Normal forms are successive levels based on functional dependencies; at least one is described correctly (1NF: atomic values, no repeating groups; 2NF: no partial dependency on part of a composite key; 3NF: no transitive dependency between non-key attributes). | names normal forms (1NF/2NF/3NF) but does not describe what any of them requires, or describes one only loosely |
| K5 | Trade-off: a highly normalised design needs more joins, which can slow some reads, so designs are sometimes deliberately denormalised for performance. | mentions that there is a cost but not what it is |

| # | Misconception (flag if stated) |
|---|---|
| M1 | The main purpose of normalisation is to make queries run faster. |
| M2 | Database normalisation means scaling numeric values into a fixed range such as 0 to 1 (confusing it with data or feature normalisation). |
| M3 | A database should always be normalised to the highest normal form possible. |

Maximum: 10 marks.

## Overfitting

*DRAFT – NOT FACULTY REVIEWED* · file `research/rubrics/overfitting.json`

| # | Key point (2 marks if clearly and correctly stated) | Partial credit (1 mark) if the student… |
|---|---|---|
| K1 | Overfitting is when a model learns the training data too closely, including its noise and random quirks, instead of the general pattern. | says the model 'memorises' the data without contrasting it with the general pattern |
| K2 | Symptom: very good performance on the training data but clearly worse performance on new, unseen data (poor generalisation). | mentions only good training accuracy or only poor test accuracy, not the gap |
| K3 | Causes: a model that is too complex for the amount of data (too many parameters or features), too little or noisy training data, or training for too long. | names one cause vaguely, e.g. 'too much training' |
| K4 | Detection: compare training error with error on a held-out validation or test set, or use cross-validation; validation error rising while training error falls is the sign. | says 'use test data' without explaining what to compare |
| K5 | Remedies: more data or data augmentation, a simpler model or fewer features, regularisation (e.g. L1/L2, dropout), early stopping, pruning. | names only one remedy, or remedies without how they help |
| K6 | Contrast with underfitting (too simple, poor on both training and test data); overfitting corresponds to high variance in the bias-variance trade-off. | mentions underfitting without explaining the difference |

| # | Misconception (flag if stated) |
|---|---|
| M1 | Overfitting means the model performs poorly on its training data (this is underfitting). |
| M2 | Higher training accuracy always means a better model, so training longer always helps. |
| M3 | Checking accuracy on the training data is enough to detect overfitting. |

Maximum: 12 marks.

## Process vs thread

*DRAFT – NOT FACULTY REVIEWED* · file `research/rubrics/process-vs-thread.json`

| # | Key point (2 marks if clearly and correctly stated) | Partial credit (1 mark) if the student… |
|---|---|---|
| K1 | A process is a program in execution with its own private address space and resources (memory, open files, etc.); it is the OS's unit of isolation. | says a process is 'a running program' without its own memory/resources |
| K2 | A thread is a unit of execution scheduled by the OS inside a process; a process has one or more threads. | says a thread is 'part of a process' without it being the unit of execution |
| K3 | Threads of one process share its code, heap/global data and open files, but each thread has its own stack, registers and program counter. | says threads share memory but not what each thread keeps for itself |
| K4 | Because threads share memory they communicate directly but need synchronisation (e.g. locks) to avoid race conditions; processes are isolated and communicate through IPC (pipes, sockets, shared memory), and one crashing usually does not crash another. | mentions either the shared-memory risk or IPC, not the contrast |
| K5 | Creating and switching between threads is cheaper than for processes, because no new address space has to be set up or switched. | says threads are 'lighter' without the reason |

| # | Misconception (flag if stated) |
|---|---|
| M1 | Each thread has its own separate memory, just like a process. |
| M2 | A thread and a process are the same thing, or a thread is a small separate program. |
| M3 | Using more threads always makes a program faster. |

Maximum: 10 marks.

## TCP vs UDP

*DRAFT – NOT FACULTY REVIEWED* · file `research/rubrics/tcp-vs-udp.json`

| # | Key point (2 marks if clearly and correctly stated) | Partial credit (1 mark) if the student… |
|---|---|---|
| K1 | Both are transport-layer protocols that run over IP and use port numbers to reach applications. | says they are 'network protocols' without the transport layer or ports |
| K2 | TCP is connection-oriented (a three-way handshake sets up a connection); UDP is connectionless and just sends datagrams. | says TCP 'connects' and UDP does not, with no detail |
| K3 | TCP gives reliable, ordered delivery using sequence numbers, acknowledgements and retransmission; UDP is best effort, with no guarantee of delivery, order or duplicate removal. | says TCP is 'reliable' and UDP is 'unreliable' without saying how or what is guaranteed |
| K4 | TCP has flow control and congestion control; UDP has neither. | mentions one of the two, or 'TCP controls the speed' vaguely |
| K5 | UDP has lower overhead (8-byte header vs at least 20 bytes, no handshake, no waiting for retransmissions), so lower latency; TCP is a byte stream, UDP keeps message boundaries. | says UDP is 'faster' or 'lighter' without the reason |
| K6 | Typical uses: TCP for the web, email and file transfer; UDP for DNS, live audio/video, VoIP and online games, where timeliness matters more than perfect delivery. | gives uses for only one protocol, or examples without the reason |

| # | Misconception (flag if stated) |
|---|---|
| M1 | UDP is less secure than TCP (neither protocol encrypts data; security comes from TLS, DTLS and similar). |
| M2 | UDP packets travel through the network faster than TCP packets. |
| M3 | TCP guarantees that data always arrives, even if the connection breaks. |

Maximum: 12 marks.

## Virtual memory

*DRAFT – NOT FACULTY REVIEWED* · file `research/rubrics/virtual-memory.json`

| # | Key point (2 marks if clearly and correctly stated) | Partial credit (1 mark) if the student… |
|---|---|---|
| K1 | Virtual memory gives each process its own private, contiguous virtual address space that is independent of the size and layout of physical RAM. | says programs 'think they have more memory' without the per-process address space |
| K2 | Virtual addresses are translated to physical addresses by the MMU (hardware) using page tables kept by the OS; a TLB caches recent translations. | mentions translation or page tables but not who does it, or the TLB alone |
| K3 | Paging: memory is divided into fixed-size pages (virtual) and frames (physical); only the pages in use need to be in RAM, the rest can stay on disk (swap / backing store), loaded on demand. | mentions pages or disk storage without explaining that only needed pages are in RAM |
| K4 | A page fault happens when a process touches a page not in RAM; the OS loads it from disk, evicting another page with a replacement policy (e.g. LRU) if RAM is full. | mentions page faults by name only |
| K5 | Benefits: programs larger than RAM and more processes can run, processes are isolated and protected from each other, and memory such as shared libraries can be shared. | names only one benefit |
| K6 | Cost: disk is far slower than RAM, so too many page faults cause thrashing, where the system spends most of its time paging. | says it 'can be slow' without the reason |

| # | Misconception (flag if stated) |
|---|---|
| M1 | Virtual memory is just the swap or page file, i.e. using the hard disk as extra RAM. |
| M2 | Virtual memory makes the computer faster or makes RAM faster. |
| M3 | Virtual addresses are the same as physical addresses, or all processes share one address space. |

Maximum: 12 marks.

## Review record

| Reviewer | Date | Rubrics reviewed | Outcome (approved / approved with changes / not approved) |
|---|---|---|---|
| | | | |
