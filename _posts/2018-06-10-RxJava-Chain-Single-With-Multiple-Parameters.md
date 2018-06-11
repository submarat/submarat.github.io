---
published: true
---
Single starting to contribute to [Trust Wallet](http://trustwalletapp.com) in 2017 I've had the opportunity touch reactive programming with RxJava. We needed to use RxJava to keep asynchronous code like network calls and db updates manageable.

To be honest it has been a steep learning curve and I quickly realized that the reactive programming was a far cry from the imperative code I was used to writing. Here's a trick I picked up recently.

RxJava has the `Observable` object at it's core and can be thought of as a stream of future events which you can `subscribe` and observe. `Single` is a special type of `Observable` which yields one result. We use `Single` often because we often want to fetch just one object.

Say you have async functions `fa()`, `fb()` and `fc(A a, B b)` which yield three singles `Single<A>`, `Single<B>` and `Single<C>`. `fc(A a, B b)` depends on the inputs from `fb()` and `fc()`. What is the best way to construct `Single<C>`? This is the best way I've found:

```
Single<A> fa() { ... }
Single<B> fb() { ... }
Single<C> fc(A a, B b) { ... }

Single<C> = fa().flatMap( a -> fb()
    		.flatMap( b -> fc(a, b) )
        )
```
