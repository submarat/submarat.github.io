---
published: false
---
[The paper](https://static.googleusercontent.com/media/research.google.com/en//archive/chubby-osdi06.pdf)

Section 1:
Chubby is used by GFS and Bigtable for storing shared metadata. Interface similar to filesystem. Based on Paxos distributed concensus algorithm.

Section 2:...

2.1  Rationale - coarse-grained locking, cells, Paxos and consensus, not-time-based locking on a file.

2.2 System Structure - master server, replicas (usually 5), master lease - time during swich master stays elected, DNS for contacting members of the group, election protocol.

2.3 Files, directories and handles - file system similar but simpler than UNIX'. ACLs stored on ephemeral nodes as files.

2.4 Locks and sequencers - lock-delay disallows a faulty client from locking a resource for an arbitary time, locking requires write-permissions, lock checking is not mandatory. Exclusive writer-mode, shared reader-mode on lock files.

2.5 Events - clients can subscribe to events. E.g. permissions change, child node added/removed - can be used to implement mirroring.

2.6 API - opaque handle to file, open(), close(), delete(), getSequencer() - similar to a file system.

2.7 Caching - 

Questions:
- What are the key use cases?
