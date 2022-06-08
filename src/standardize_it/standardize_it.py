__author__ = 'William Hegedusich'
__date__ = '10/1/2021'
__all__ = ['Standardizer',
           ]

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Standardizer:
    def __init__(self, standards: List, ng_len: Tuple = (2, 2), **kwargs: Any) -> None:
        """
        :param standards: list of standardized strings to be used in the function
        :param ng_len: min and max length of n-grams for the vocabulary
        :keyword threshold: Cosine similarity threshold that sets the cutoff limit for
                            determining if words are similar enough.
        :keyword analyzer: Analyzer to use to generate n-grams: One of {'word', 'char', 'char-wb'}, default 'char'
        """

        if len(standards) == 0:
            raise ValueError("List of standard strings cannot be empty.")
        if len(ng_len) > 2:
            raise ValueError("N-Gram range must only contain 2 elements.")
        if 0 in ng_len:
            raise ValueError("N-Gram range cannot contain zeroes.")
        if ng_len[0] > ng_len[1]:
            raise ValueError("Max N-Gram length cannot be larger than Min N-Gram length.")

        self._analyzer = kwargs.get('analyzer', 'char')
        self._input_as_vectors = {}
        self._last_results = None
        self._new_strings = None
        self._ng_len = ng_len
        self._questionable = None
        self._raw = None
        self._standards = standards

        # Cosine Similarity threshold for determining if a new string is accurate or not
        self._threshold = kwargs.get("threshold", 0.45)

        self._vectorizer = CountVectorizer(analyzer=self._analyzer, ngram_range=self._ng_len)

        # Make vectors for target make names
        self._standard_vectors = dict(
            zip(self._standards, self._fit_cv())
        )

    @property
    def input_as_vectors(self) -> Dict:
        """
        Getter method for input values and their corresponding vectors as transformed by the vectorizer
        :return: input as vectors
        """
        if self._input_as_vectors == {}:
            raise ValueError("No input has been supplied yet.")

        return self._input_as_vectors

    @property
    def last_results(self) -> Dict:
        """
        Getter method for
        :return: self.__last_results
        """
        if not self._last_results:
            raise ValueError
        return self._last_results

    @property
    def new_strings(self) -> List:
        """
        Getter method for new strings
        :return: List of the strings most similar to the original string
        """
        if not self._new_strings:
            raise ValueError
        return self._new_strings

    @property
    def ng_len(self) -> Tuple:
        """
        Getter method for NG Len
        :return: N-Gram length range
        """
        return self._ng_len

    @property
    def questionable(self) -> Dict:
        """
        Getter method for questionable matches
        :return: list of new strings with cosine similarities below the established threshold
        """
        if self._questionable is None:
            raise ValueError("Questionable is 'None'")

        return self._questionable

    @property
    def raw(self) -> List:
        """
        Getter function for raw input
        :return: List of raw inputs
        """
        if self._raw is None:
            raise ValueError
        return self._raw

    @property
    def standards(self) -> List:
        """
        Getter method for standards
        :return: List of standard strings
        """
        return self._standards

    @standards.setter
    def standards(self, stds: List):
        """
        Update list of standards and refit Vectorizer
        :param stds: New standards
        :return:
        """
        self._standards = stds
        self._standard_vectors = dict(
            zip(self._standards, self._fit_cv())
        )

    @property
    def threshold(self) -> int:
        return self._threshold

    @threshold.setter
    def threshold(self, t: int) -> None:
        self._threshold = t

    @property
    def standard_vectors(self) -> Dict:
        """
        Getter method for standardized values as vectors
        :return: standards as vectors
        """
        return self._standard_vectors

    def compare(self) -> List:
        """
        Compare raw inputs to new strings
        :return: List of tuples
        """
        if self._raw is None:
            raise ValueError("Raw values must be provided and converted before comparing.")
        elif not self._new_strings:
            raise ValueError("No new strings available for comparison.")
        elif (lraw := len(self._raw)) != (lnew := len(self._new_strings)):
            raise ValueError(f"Cannot compare raw list of length {lraw} with new string list of length {lnew}.")

        return list(zip(self._raw, self._new_strings))

    def get_related(
            self,
            get_from: Literal['new', 'raw'],
            raw_val: Union[int, str, List[Union[int, str]]],
            n: Optional[int] = 1
    ) -> Union[str, List[str]]:
        """
        Method for getting a specific new string from the raw input
        :param get_from: Indicate whether to get new strings from raw input, or raw input from new string
        :param raw_val: Can be either a raw string, integer index of the raw string, or a list of strings or indices
        :param n: Number of new_string candidates to return; defaults to 1, i.e. the most similar string
        :return: New string(s) corresponding to the specified raw value(s)
        """

        if isinstance(raw_val, str):
            newstrings = self._get_by_str(get_from=get_from, s=raw_val, n=n)

        elif isinstance(raw_val, int):
            newstrings = self._get_by_int(get_from=get_from, i=raw_val, n=n)

        else:
            if any([not isinstance(v, (int, str)) for v in raw_val]):
                raise TypeError("If `raw_val` is a list, it can only contain types `int` and/or `str`.")
            newstrings = [self._get_by_str(get_from=get_from,
                                           s=v,
                                           n=n)
                          if isinstance(v, str) else
                          self._get_by_int(get_from=get_from,
                                           i=v,
                                           n=n)
                          for v in raw_val]

        return newstrings

    def standardize_it(self, raw: List, **kwargs: Any) -> None:
        """
        Takes a list of strings and standardizes them using cosine similarities
        :param raw: List of car names
        :return: None
        """
        if len(raw) == 0:
            raise ValueError("Argument 'raw' cannot' be empty.")

        if any([not isinstance(i, str) for i in raw]):
            raise TypeError("Argument 'raw' can only take a list comprising strings;"
                            "please check for other dtypes in the list.")

        self._raw = raw
        self._threshold = kwargs.get('threshold', self._threshold)
        self._last_results = {}

        # Make vectors for the input make names
        self._input_as_vectors = dict(zip(self._raw, self._vectorizer.transform(raw).toarray()))

        # Calculate the cosine similarities
        for val in raw:
            if val in self._last_results.keys():
                # If sims are already known for a value, we can skip it and use previous sims for replacement
                continue
            elif val in self._standards:
                self._last_results[val] = {val: 1.00}
                continue
            else:
                """
                The count vectorizer only makes one vector for each raw string, even if it appears more than once.
                As such we need to access the vector in input_as_vectors using the raw name as the key,
                and pass that vector as the argument. Otherwise each misspelling only gets transformed once.
                """
                sims = self._calc_cosine_sim(word=self._input_as_vectors[val])
                self._last_results[val] = sims

        # Call function to create a list of the most similar standard strings
        self._most_similar()

    def _calc_cosine_sim(self, word: np.ndarray) -> Dict:
        """
            Calculates the cosine similarity of 'word' against all members of 'targets'
            :param word: String as currently spelled - to be corrected
            :return: dict of similarities, with standard string as key and the similarity to the raw string as the value
            """
        sims = []  # List of similarities to the word in question

        for tgt in self._standard_vectors.values():
            # Use index to access the actual numeric value instead of storing it as `np.ndarray(['similarity'])`
            sims.append(cosine_similarity([word], [tgt])[0][0])

        # Make dict of target makes, and the cosine similarity with each
        target_sims = dict(zip(self._standards, sims))

        # Reorder dict to go from highest to lowest similarity
        target_sims = {k: v for k, v in sorted(target_sims.items(), key=lambda x: x[1], reverse=True)}

        return target_sims

    def _fit_cv(self) -> np.ndarray:
        """
        Fit CountVectorizer to list of standards
        :return: Vectorized standards as an array
        """
        return self._vectorizer.fit_transform(self._standards).toarray()

    def _get_by_int(self, get_from: str, i: int, n: Optional[int]):
        """
        Get raw string using index provided, then call get_by_str to return list of strings
        :param get_from: indicates if getting related from new strings or raw input
        :param i: Index of raw value
        :param n: Number of elements to return
        :return: string or List of all new strings corresponding to that raw value
        """
        if get_from == 'raw':  # If trying to get new strings from raw inputs, need to index into raw input strings
            return self._get_by_str(get_from=get_from, s=self._raw[i], n=n)
        else:  # `by` must necessarily be `new` here since it was type-checked against a Literal
            return self._get_by_str(get_from=get_from, s=self._new_strings[i], n=n)

    def _get_by_str(self, get_from: str, s: str, n: Optional[int]) -> Union[str, List]:
        """
        Get a value from new strings by the string
        :param get_from: indicates if getting related from new strings or raw input
        :param s: String to use as getter reference for other values
        :param n: Number of elements to return, defaults to `all`
        :return: string or List of all new strings corresponding to that input value
        """
        if get_from == 'raw':  # If trying to get new strings from raw inputs, get value from __last_results
            res = list(self._last_results[s].keys())
        else:  # by must necessarily be `new` here since it was type-checked against a Literal
            """
            Iterate through results
            Get the values (i.e. the dict of sims) for each key (i.e. raw value) in __last_results
            If the first key (i.e. the most similar) matches argument `s`, return that top-level key (i.e. raw string)
            This has the effect of returning all of the raw strings that were standardized to the input `new string` 
            """
            res = [raw for raw, sims in self._last_results.items() if list(sims.keys())[0] == s]

        return res[0] if n == 1 else res[:n]  # Return top `n` strings

    def _most_similar(self) -> None:
        """
        Returns the first standardized value (i.e. most similar) for each raw input string
        :return: List of standardized strings most aligned with raw input
        """
        self._new_strings = []
        self._questionable = {}

        for val in self._raw:
            # For each raw value, get the most similar string and its cosine similarity score
            top_result = list(self._last_results[val].items())[0]

            if top_result[1] <= self._threshold:
                # If the most similar score is below the user-defined threshold, append it to the `questionable` dict
                self._questionable[val] = top_result

            self._new_strings.append(top_result[0])  # Append most similar string to new_strings

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._get_by_str(get_from='raw', s=item, n=None)
        elif isinstance(item, int):
            return self._get_by_int(get_from='raw', i=item, n=None)
        else:
            raise TypeError(f"Type {type(item)} is not a valid indexer.")
