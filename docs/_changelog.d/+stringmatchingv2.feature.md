Very large memory savings from an overhaul of the string matching process.
The new version should also be faster in general, but the priority was memory rather than speed (since previously, this step accounted for the majority of kazu's memory usage but only a fraction of its runtime)
