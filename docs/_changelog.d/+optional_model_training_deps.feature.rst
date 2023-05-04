Moved model training specific dependencies to an optional dependency group.
In particular, this is of value because the seqeval dependency doesn't distribute
a wheel, only an sdist in a legacy manner, which broke kazu installation in
environments requiring proxies.
