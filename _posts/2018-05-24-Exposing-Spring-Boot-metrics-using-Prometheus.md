---
published: true
---
Adding metrics to your microservices is now a non-negotiable. Luckily Spring Boot has built in capabilities for this using micrometer. Prometheus is a timeseries database that makes it easy to analyze the raw numbers streaming from your server.

I faced some trouble in exposing metrics but in the end brute-forced to this combination.

## Modules

In gradle file:

{% highlight bash %}
spring-boot-starter-web: 2.0.0.RELEASE
spring-boot-starter-actuator: 2.0.2.RELEASE
micrometer-registry-prometheus: 1.0.1
{% highlight bash %}

## Properties

properties.yaml:
```
management:
    endpoints:
        web:
            exposure:
                include: prometheus,metrics,info,health
```

## Test


> http GET http://localhost:8080/actuator/prometheus
