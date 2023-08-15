Make SpanFinder return found spans directly, rather than having to access `.closed_spans` after calling, which is easier. Note that `.closed_spans` remains, so this is backwards-compatible.
