---
published: false
---
[Kafka paper](http://notes.stephenholiday.com/Kafka.pdf)

Questions:
- Why is having no master node advantageous?
- How is Zookeeper similar and different to Kafka? Is it being used as a subsystem?

Zookeeper is like a distributed file system. It is used by Kafka for consensus among brokers and consumers.

- How are performance gains achieved? How does performance compare against similar systems?

Stateless broker - broker doesn't keep the state for each reader, clients track that themselves. To get around deleting entries prior to everything being consumed Kafka uses an (7-day) SLA after which data is cleared.

Simple storage - data is stored as blocks of bytes (1GB), clients are responsible for choosing their own serialization formats.

Efficient transfer - uses Linux/Unix sendFile() method for avoiding unnecessary kernel and system operations.

- Is the broker stateless? What is the benefit?
- What does CRC mean?
- What is Avro and schema evolution?
- What are the key components of Kafka?
- What is Kafka good and not good for?

Sections:
3. Kafka architecture and design principles

There are producers <-> brokers <-> consumers. Messages are published in topics to multiple brokers. Consumers are given iterators to consume streams of data. Iterators do not terminate if there are no more messages but rather block.

3.1 Efficiency on a single partition - Simple storage, efficient transfer, stateless broker

3.2 Distributed coordination - Zookeeper is used to rebalance load and reach consensus without a master server.

3.3 Delivery guaranees - at-least once deliver, client must implement own deduplication, order is ensured from same partition but not from multiple, per-message CRC used to prevent corruption, broker storage permanently going down means messages are lost.

4 Kafka at LinkedIn