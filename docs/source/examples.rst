standardize-it examples
=======================

Assume we are presented with a list of vehicle models like this::

    raw_makes = [
        "CHEVR",
        'NISSA",
        "HYUND",
        "TOYOT",
        "CHRYS",
        "MERCE",
        "SUBAR",
        "VOLKS",
        "FORD",
        "FORD TRUCK",
        "CHERVOLET",
        "CHEVY",
        "CHEVY TRUCKS",
        "NISS",
        "MAZD",
        "CADI"
    ]

and presume this list is at least 10-20 times as long, and the number are variations are too many to list here.

It's easy enough to look at these and know what they SHOULD be, and it's also easy enough to go get a list of all auto makes.
However, manually doing this task is tedious and potentially unreliable.

We can solve this by utilizing the ``Standardizer`` class::

    from standardize_it import Standardizer

    standards = [
        "CHEVROLET",
        "NISSAN",
        "HYUNDAI",
        "TOYOTA",
        "NISSAN",
        "CHRYSLER",
        "MERCEDES-BENZ",
        "SUBARU",
        "VOLKSWAGEN",
        "FORD",
        "MAZDA",
        "CADILLAC"
    ]

    standardizer = Standardizer(standards)
    standardizer.standardize_it(raw_makes)

    print(standardizer.new_strings)

This is the simplest implementation of the ``Standardizer`` class. The Standardizer instance can be further configured by
setting n_gram lengths, a cutoff threshold, and different analyzers for the n_grams::

    from standardize_it import Standardizer

    standardizer = Standardizer(standards, ng_len=(1,5), threshold=0.5, analyzer="char-wb")
    standardizer.standardize_it(raw_makes)

    print(standardizer.new_strings)
