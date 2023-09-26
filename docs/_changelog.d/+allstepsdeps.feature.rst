Slimmed down base dependencies by removing dependencies for steps not in the base pipeline.
These can be added back in manually in user projects, or use the new `kazu[all_steps]` dependency
group to install dependencies for all steps as before. The docs reflect this, and informative errors
are raised when trying to use these steps when dependencies aren't installed.
