---
published: false
---
[The paper](https://static.googleusercontent.com/media/research.google.com/en//archive/chubby-osdi06.pdf)

Section 1:
Chubby is used by GFS and Bigtable for storing shared metadata. Interface similar to filesystem. Based on Paxos distributed concensus algorithm.

Section 2:...

2.1  Rationale - coarse-grained locking, cells, Paxos and consensus, not-time-based locking on a file.

2.2 System Structure - master server, replicas (usually 5), master lease - time during swich master stays elected, DNS for contacting members of the group, election protocol.

2.3 Files, directories and handles - 

Questions:
- What are the key use cases?
