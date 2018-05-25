---
published: true
---
I was asked this question at an onsite once: Given a non-empty array of integers, return the k most frequent elements.

Example
Input [1, 2, 2, 4, 3, 1, 5], k = 2
Output: [1, 2]

## Brute force

A relatively brute-force solution is to create a frequency hashmap, sort the results by frequency and return the top k results in the sequence. Runtime: `O(nlogn)` due to sorting all entries.

## Optimization 1: use max heap

Conceptually we don't care for sorting all entries in the frequency map, when we need fewer than n results. So instead we could use max heap which performs comparisons based on frequency and take k values from it. Considering than we can construct the heap in `O(n)` time, it would require `O(k)` retrievals each costing `O(logn)`. So resulting complexity is `O(klogn)`. This is an improvement because k <= n, but it turns out that we can do better.

## Optimization 2: bucket sort

We can trade space for time complexity by using the knowledge that the maximal frequency of any element will be the length of the list. We can allocate an array to store lists of keys at position corresponding to the frequency of those elements. This way we can scan across our buckets in reverse to obtain k most frequent elemnts

## K most frequent in stream

This problem can be extended a system design question where the array cannot possibly fit into the memory of a single machine. There are two approaches we can take then: use optimization 1 + shard and lossy counting.

## Sharding

We can instantiate N machines and shard the items in the stream using the modulo operation i.e. hash(item) % N = X, X correspoding to the index of the machine. To produce k largest, we query each instance and merge the sorted lists items.

## Lossy counting

Divide the stream into frames, for each frame, count frequences and decrease by 1 then move to next frame. We evict items whose frequencies drop to 0. The idea being that items with low frequency will have low probability of staying in the frequency map. This is a great improvement over the previous approach because we can store the results on one machine. Though this approach doesn't seem like it'll work for a stream of homogenous frequencies.

## References
(Top k elements)[https://zpjiang.me/2017/11/13/top-k-elementes-system-design/]
