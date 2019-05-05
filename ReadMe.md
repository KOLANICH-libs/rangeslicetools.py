rangeslicetools.py [![Unlicensed work](https://raw.githubusercontent.com/unlicense/unlicense.org/master/static/favicon.png)](https://unlicense.org/)
==================
~~[wheel (GHA via nightly.link)](https://nightly.link/KOLANICH-libs/rangeslicetools.py/workflows/CI/master/rangeslicetools-0.CI-py3-none-any.whl)~~
~~[wheel (GitLab)](https://gitlab.com/KOLANICH/rangeslicetools.py/-/jobs/artifacts/master/raw/dist/rangeslicetools-0.CI-py3-none-any.whl?job=build)~~
~~[![GitLab Build Status](https://gitlab.com/KOLANICH/rangeslicetools.py/badges/master/pipeline.svg)](https://gitlab.com/KOLANICH/rangeslicetools.py/pipelines/master/latest)~~
~~![GitLab Coverage](https://gitlab.com/KOLANICH/rangeslicetools.py/badges/master/coverage.svg)~~
~~[![GitHub Actions CI](https://github.com/KOLANICH-libs/rangeslicetools.py/workflows/CI/badge.svg)](https://github.com/KOLANICH-libs/rangeslicetools.py/actions/)~~
![N∅ dependencies](https://shields.io/badge/-N%E2%88%85_deps!-0F0)
[![Code style: antiflash](https://img.shields.io/badge/code%20style-antiflash-FFF.svg)](https://github.com/KOLANICH-tools/antiflash.py)

This is a library to manipulate python `range` and `slice` objects. The objects of these classes have the same internal structure but a bit different semantics and set of available methods. Unfortunately these objects include no methods to manipulate them and unfortunately they cannot be subclassed.

So I have implemented a set of functions to manipulate these objects. Their names follow the following conventions:

* All these **functions** names begin from `s` which stands there for `slice`, even though they will work for ranges too.

* If a name ends with `_`, it is a generator, otherwise it returns a `list`.

**WARNING: FOR NEGATIVE-DIRECTED `slice`s/`range`s `step` is MANDATORY. It is BY DESIGN of python and we follow this convention too. Always set `step` for all the ranges if you may deal with negative-directed ones.**

For the info on usage see the docstrings and tests. And READ the source code, it is SMALL ENOUGH.

Features
--------
Notation and terms:

* For briefness we `r = range` and `s = slice`
* When we say `range`, it also works for a `slice` and in the opposite direction too.

Conventions:

* When we say `range` or `slice`, it usually works also for a sequence of them. See type annotations to check if a specific function supports sequences of ranges.
* There may be undefined behavior (UB):

	* negative-directed ranges without negative `step` is always UB;
	* non-integer numbers usage is always UB;;
	* operations on the ranges having different `abs(step)`;
	* empty ranges (ranges of zero length) produced **may** (but not guaranteed) be eliminated;
	* operations on the ranges having opposite direction may be UB. It should be stated in docstrings if it is the case.

* basic operations

	* type conversion:
		* `sAny2Type(s(1, 10), r) -> r(1, 10)` `sAny2Type(r(1, 10), s) -> s(1, 10)`
		* `range2slice` and `slice2range` do the same.

	* get a length of a range `slen(s(2, 4)) -> 2`. For usual ranges just `len` works, but our func works also for slices and seqs.
	* get direction (a director vector in fact) of a slice `sdir(r(10, 1, -2)) -> -1`
	* reverse direction of a slice: `srev(r(0, 10)) -> r(9, -1, -1)`
	* make 2 `slice`s of the same direction: `sdirect(r(25, 5, -5), s(1, 10)) -> slice(9, 0, -1)`
	* make a slice positive-directed: `snormalize(r(25, 5, -5)) -> range(10, 30, 5)`

* checking conditions about ranges:

	* check if one range is fully within another range `swithin(r(0, 10), r(1, 5)) -> true`
	* check if one range is overlaps another range `soverlaps(r(0, 10), r(2, -6, -1)) -> true`

* splitting

	* sprit a range at certain points: `ssplit(r(5, 13), (7, 8, 12)) -> [r(5, 7), r(7, 8), r(8, 12), r(12, 13)]`
	* split a range into pieces of certain lengths `soffset_split(r(5, 13), (2, 3, 7)) -> [r(5, 7), r(7, 8), r(8, 12), r(12, 13)]`
	* split a range into pieces of a certain length `schunks(r(5, 13), 3) -> [r(5, 8), r(8, 11), r(11, 13)]`
	* split multiple sequences of ranges of the same total length into the chunks of equal length, in other words - align split points of all the sequence - see the docs for `salign` function.

* join/merge **adjacent** (non-overlapping!) ranges into one: `sjoin([r(0, 8), r(8, 9), r(9, 10), r(12, 15)]) -> [r(0, 10), r(12, 15)]`

* set operations

	* compute a diff of 2 ranges: `sdiff`
	* subtract 2 ranges: `ssub(r(1, 10), r(5, -10, -1)) -> [r(6, 10)]`
	* union 2 ranges: `sunion(r(1, 10), r(7, 20)) -> [r(1, 20)]` 

* intersections querying via a [range tree](https://en.wikipedia.org/wiki/Range_tree)
* remapping via a `SliceSequence`
* visualization


Examples
--------
* https://codeberg.org/kaitaiStructCompile/Endianness.py
* tests


Similar projects
----------------

* [intervaltree](https://github.com/chaimleib/intervaltree)
* [rangetree](https://github.com/nanobit/rangetree)
