# Kazu-JVM - Biomedical NLP

Some valuable BioNLP algorithms are implemented in JVM languages (e.g. SETH, Opsin). In order to make these
accessible to python, we use Py4J to call them from a Kazu Step instance.

# Compiling the JVM
In order to make such algorithms available to Py4J, we need to compile a fatjar, which can then be placed on the 
classpath of the JVM process started by Py4J.

We use the gradle wrapper concept
https://docs.gradle.org/current/userguide/gradle_wrapper.html

```shell
./gradlew shadowJar

```

# Tests
Tests can be run as follows:
```shell
./gradlew test

```
