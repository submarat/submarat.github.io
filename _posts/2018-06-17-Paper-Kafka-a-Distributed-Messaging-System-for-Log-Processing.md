---
published: true
---
This post summarizes the [Kafka paper](http://notes.stephenholiday.com/Kafka.pdf) and my thoughts about it.

# Architecture

The basic components of Kafka are producers, brokers, consumers. Messages are submitted by producers to brokers via topics, consumers read messages from topics. Each broker is responsible for a single partition of the data. Brokers are coordinated in a distributed fashion using ZooKeeper.

# Reliability

Messages have CRC (cyclic redundancy check) included. If the check fails, the message is dropped. Since the publishing of the paper brokers/partitions can have a tunable level of redundancy and replication for increased reliability.  

# Performance

One of the key advantages of Kafka over competing systems is performance. In tests, Kafka performed an order of magnitude faster than both Active MQ and Rabbit MQ. This performance gain was due to several specializations:
 * stateless brokers - don't monitor consumer state. Consumer is responsible for tracking its own offset.
 * `sendfile` function of Linux kernel - more efficient that using `read` and `write`
 * efficient storage - messages are stored as byte data, are not given unique ids. A block offset is used instead
 * builk send - producers can send messages in bulk reducing network overhead

# Disadvantages

The paper didn't highlight too many. Order of messages can be ensured on a single partition but when reading between many partitions. There is a x-day SLA within which messages are persisted. Consumers are responsible for their own deduptication and serialization scheme (Avro was used by LinkedIn). 

# Commercial use

Kafka has been successfully used at Linkedin and multi other companies. Though the origin use case is web service log data, it can be used wherever a messaging queue is required. During a conversation with some folks over at Confluent - the commercial Kafka offering from three of its founders - I learned that there is a growing adoption of Kafka and customers are willing to pay for increased reliabilty, monitoring, connectors, in-stream data query and support.

# Discussion

This was a well-written paper though I found it somewhat terse. It would have been good to see more extensive testing of the system. The authors made it clear that Kafka was not a panacea but did not give too many scenarios where the system would underperform.