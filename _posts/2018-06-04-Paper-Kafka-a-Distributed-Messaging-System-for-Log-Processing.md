---
published: false
---
http://notes.stephenholiday.com/Kafka.pdf

Questions:
- Why is having no master node advantageous?
- How is Zookeeper similar and different to Kafka? Is it being used as a subsystem?
- How are performance gains achieved? How does performance compare against similar systems?
- Is the broker stateless? What is the benefit?
- What does CRC mean?
- What is Avro and schema evolution?
- What are the key components of Kafka?
- What is Kafka good and not good for?

Sections:
3. Kafka architecture and design principles

There are producers <-> brokers <-> consumers. Messages are published in topics to multiple brokers. Consumers are given iterators to consume streams of data. Iterators do not terminate if there are no more messages but rather block.