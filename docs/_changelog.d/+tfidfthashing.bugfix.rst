Fix incorrect caching behaviour of Index TfidfVectorizer builds.
This meant they got rebuilt every time, which meant in turn that the cache and therefore the model pack size grew after use.
