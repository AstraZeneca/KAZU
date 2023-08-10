Kazu as a Library
===================

We understand that many use cases for NER and Linking tend to come as part of a larger data processing
pipeline, and thus it makes sense to use Kazu as a library of code. This is generally easy to do, with
the following caveats:


Dependency conflicts
-----------------------

`Capping dependencies for a library's requirements is a controversial aspect
of development <https://iscinumpy.dev/post/bound-version-constraints/>`_.

We try to keep kazu as flexible as possible with regard to dependencies. This means we don't cap
dependencies, and this can sometimes cause unforeseen errors. To this end, we test kazu with the
latest version of each of its dependencies as descibed in the ``pyproject.toml`` file. If you
suspect you are having dependency clash issues, you can view the dependencies a given Kazu
model pack was tested with via the ``tested_dependencies.txt`` file (located at the top
level of a model pack). Try installing the version of the problematic dependency listed here.
