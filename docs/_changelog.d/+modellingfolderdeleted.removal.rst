We deleted the 'modelling' element of the module structure, as this didn't add anything semantically, but led to longer imports.
This does mean any imports from these directories in your code will need to be rewritten - fortunately this is a simple 'find and delete' with 'modelling.' within your imports.
