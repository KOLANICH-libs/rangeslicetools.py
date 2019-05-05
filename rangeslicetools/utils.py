import heapq
import itertools
import typing
from collections.abc import Sequence
from functools import wraps

__all__ = ("SliceRangeT", "SliceRangeTypeT", "SliceRangeSeqT", "SliceRangeListT", "sAny2Type", "range2slice", "slice2range", "slen", "sdir", "svec", "srev", "sdirect", "snormalize", "ssplit_1_", "ssplit_1", "ssplit_", "ssplit", "schunks_", "schunks", "soffset_split_", "soffset_split", "sjoin_", "swithin", "soverlaps", "teeSliceSequences", "salign_", "sPointIn", "ssegments_", "ssegments", "shull")

isInstArg = (range, slice)
SliceRangeT = typing.Union[isInstArg]
SliceRangeTypeT = typing.Union[tuple(typing.Type[el] for el in isInstArg)]
SliceRangeSeqT = typing.Iterable[SliceRangeT]
SliceRangeListT = typing.Sequence[SliceRangeT]
SliceRangeOptListT = typing.Union[SliceRangeT, SliceRangeListT]


def _getStepForComputation(slc: SliceRangeT) -> int:
	"""Returns a `step` that is a number"""
	if slc.step is not None:
		return slc.step

	if slc.start <= slc.stop:
		return 1

	raise ValueError("start < end, so if step is not explicitly defined, it is undefined! Setup the step explicitly (you would likely need -1)!")


def sign(n: int) -> int:
	"""Signum func FOR OUR PURPOSES"""
	if n is None or n >= 0:
		return 1

	return -1


def _scollapse(slc: SliceRangeOptListT) -> SliceRangeOptListT:
	"""Collapses a sequence of ranges into a range, if it contains only a 1 range"""
	if not isinstance(slc, isInstArg) and len(slc) == 1:
		return slc[0]
	return slc


def sAny2Type(rng: SliceRangeT, tp: SliceRangeTypeT) -> SliceRangeT:
	"""Creates a new /range/slice with needed type"""
	return tp(rng.start, rng.stop, _getStepForComputation(rng))


def range2slice(rng: SliceRangeT) -> slice:
	"""Clones into a slice."""
	return sAny2Type(rng, slice)


def slice2range(slc: SliceRangeT) -> range:
	"""Clones into a range."""
	return sAny2Type(slc, range)


def _slen(slc: SliceRangeT) -> int:
	return len(slice2range(slc))


def slen(slcs: SliceRangeSeqT) -> int:
	"""Returns length of a range/slice."""
	if isinstance(slcs, isInstArg):
		return _slen(slcs)
	total = 0
	for s in slcs:
		total += _slen(s)
	return total


def sdir(slc: SliceRangeT) -> int:
	"""Returns director of a range/slice."""
	return sign(slc.stop - slc.start)


def svec(slc: SliceRangeT) -> int:
	return sdir(slc) * slen(slc)


def srev(slc: SliceRangeT) -> SliceRangeT:
	"""Reverses direction of a range/slice."""
	step = _getStepForComputation(slc)
	newStep = -1 * step
	assert isinstance(slc, range) or newStep >= -1, "Negative-directed slices with `step`s other -1 don't work!"
	return slc.__class__(slc.stop - step, slc.start - step, newStep)


def _isNegative(slcs: SliceRangeListT) -> typing.Iterable[bool]:
	return slcs.__class__(el.stop < el.start for el in slcs)


def _sdirect(donorNegative: bool, acceptor: SliceRangeOptListT) -> SliceRangeOptListT:
	if not isinstance(acceptor, isInstArg):
		if not isinstance(donorNegative, bool):
			return acceptor.__class__(_sdirect(*el) for el in zip(donorNegative, acceptor))

		return acceptor.__class__(_sdirect(donorNegative, el) for el in acceptor)

	if donorNegative != (acceptor.stop < acceptor.start):
		return srev(acceptor)

	return acceptor


def sPointIn(s: SliceRangeT, pt: int) -> bool:
	#return (((s.step is None or s.step > 0) and s.start <= pt < s.stop) or (s.start >= pt > s.stop))
	return pt in slice2range(s)


def snormalize(slc: SliceRangeOptListT) -> SliceRangeOptListT:
	"""Returns range/slice that points forward. If the range is positive-directed with the step 1, removes the step."""
	res = _sdirect(False, slc)
	if isinstance(res, isInstArg):
		return sAny2Type(res, slc.__class__)

	return res.__class__(sAny2Type(el, el.__class__) for el in res)


def sdirect(donor: SliceRangeT, acceptor: SliceRangeT) -> SliceRangeT:
	"""Makes direction of an `acceptor` the same as a direction of a `donor."""
	return _sdirect(donor.stop < donor.start, acceptor)


class InBandSignal:
	__slots__ = ()


newMacroGroup = InBandSignal()


def _createWrappedWithnewMacroGroup(f: typing.Callable) -> typing.Callable:
	@wraps(f)
	def f1(*args, **kwargs):
		bigRes = []
		res = []
		secCtor = kwargs.get("_secCtor", tuple)

		def genericAppend():
			nonlocal res
			if len(res) == 1:
				res = res[0]
			else:
				res = secCtor(res)

			bigRes.append(res)
			res = []

		for el in f(*args, **kwargs):
			#ic(el)
			if el is not newMacroGroup:
				res.append(el)
			else:
				genericAppend()

		if res:
			genericAppend()

		bigRes = secCtor(bigRes)

		return bigRes

	f1.__annotations__["return"] = typing.Iterable[SliceRangeOptListT]
	return f1


def ssplit_1_(slc: SliceRangeT, splitPts: typing.Union[int, typing.Iterable[int]]) -> SliceRangeSeqT:
	"""Splits the slices by split points, which are ABSOLUTE POSITIONS OF POINTS on axis."""
	tp = slc.__class__
	if isinstance(splitPts, int):
		splitPts = (splitPts,)
	for p in splitPts:
		if p != slc.start:
			yield tp(slc.start, p, slc.step)
		yield newMacroGroup
		slc = tp(p, slc.stop, slc.step)
	yield slc


ssplit_1 = _createWrappedWithnewMacroGroup(ssplit_1_)


def ssplit_(slc: SliceRangeSeqT, splitPts: typing.Iterable[int]) -> SliceRangeSeqT:
	"""Splits the slices by split points, which are ABSOLUTE POSITIONS OF POINTS on axis."""

	if isinstance(slc, isInstArg):
		slc = (slc,)
	if isinstance(splitPts, int):
		splitPts = (splitPts,)

	splitPts = iter(splitPts)

	try:
		pt = next(splitPts)
	except StopIteration:
		yield from slc
		return

	pts2split = []

	for s in slc:
		while pt is not None and sPointIn(s, pt):
			pts2split.append(pt)
			try:
				pt = next(splitPts)
			except StopIteration:
				pt = None

		if pts2split:
			yield from ssplit_1_(s, pts2split)
			pts2split = []
		else:
			yield s


ssplit = _createWrappedWithnewMacroGroup(ssplit_)


def schunks_(slc: SliceRangeT, chunkLen: int) -> SliceRangeSeqT:
	"""Splits the slice into slices of length `chunkLen` (which is in `slc.step`s!!!)"""
	cl = chunkLen * _getStepForComputation(slc)
	return ssplit_(slc, range(slc.start + cl, slc.stop, cl))


schunks = _createWrappedWithnewMacroGroup(schunks_)


def soffset_split_(slc: SliceRangeSeqT, splitPts: typing.Iterable[int]) -> SliceRangeSeqT:
	"""Splits the slices by split points, which are OFFSETS FROM RANGE BEGINNING."""
	if isinstance(slc, isInstArg):
		slc = (slc,)
	if isinstance(splitPts, int):
		splitPts = (splitPts,)

	splitPts = iter(splitPts)

	try:
		pt = next(splitPts)
	except StopIteration:
		yield from slc
		return

	cumLen = 0
	cumLenPrev = None
	pts2split = []

	for s in slc:
		cumLenPrev = cumLen
		cumLen += slen(s)
		while pt is not None and cumLenPrev <= pt < cumLen:
			pts2split.append(s.start + (pt - cumLenPrev) * _getStepForComputation(s))
			try:
				pt = next(splitPts)
			except StopIteration:
				pt = None

		if pts2split:
			yield from ssplit_1_(s, pts2split)
			pts2split = []
		else:
			yield s


soffset_split = _createWrappedWithnewMacroGroup(soffset_split_)


def _posHull(first: SliceRangeT, slcs: SliceRangeSeqT) -> SliceRangeT:
	mi, ma = first.start, first.stop

	for slc in slcs:
		mi = min(mi, slc.start)
		ma = max(slc.stop, ma)

	return first.__class__(mi, ma, first.step)


def _negHull(first: SliceRangeT, slcs: SliceRangeSeqT) -> SliceRangeT:
	ma, mi = first.start, first.stop

	for slc in slcs:
		mi = min(mi, slc.stop)
		ma = max(slc.start, ma)

	return first.__class__(ma, mi, first.step)


def shull(slcs: SliceRangeSeqT) -> SliceRangeT:
	"""Returns the range covering all the ranges provided. Every item must be of the same direction!
	See also `sunion`, which is slower, but takes into account direction.
	"""

	slcs = iter(slcs)
	first = next(slcs)

	if first.start < first.stop:
		return _posHull(first, slcs)

	return _negHull(first, slcs)


def sjoin_(slcs: SliceRangeSeqT) -> SliceRangeSeqT:
	"""Merges adjacent or overlapped ranges. All the ranges must be of the same direction. If the direction is negative, the sequence MUST be reversed! The sequence MUST be sorted. The type is taken from the type of the first range in the input."""
	slcs = iter(slcs)
	wholeDir = 0
	while not wholeDir:
		try:
			prevSlc = next(slcs)
		except StopIteration:
			return
		wholeDir = sdir(prevSlc)
	wholeDir = wholeDir > 0  # type: bool
	tp = prevSlc.__class__

	for s in slcs:
		#assert (prevSlc.start <= prevSlc.stop) == (s.start <= s.stop)
		if prevSlc.step == s.step:
			if s.start == prevSlc.stop:
				prevSlc = tp(prevSlc.start, s.stop, prevSlc.step)
			else:
				curDir = prevSlc.start <= s.start
				if (swithin(prevSlc, s) or swithin(s, prevSlc)) or curDir == wholeDir and soverlaps(prevSlc, s):
					prevSlc = shull((prevSlc, s))
				else:
					yield prevSlc
					prevSlc = s
		else:
			yield prevSlc
			prevSlc = s
	yield prevSlc


def swithin(haystack: SliceRangeT, needle: SliceRangeT) -> bool:
	"""Answers if needle is fully within haystack (including boundaries)."""
	hsn = snormalize(haystack)
	nn = snormalize(needle)
	return _swithin(hsn, nn)


def soverlaps(haystack: SliceRangeT, needle: SliceRangeT) -> bool:
	"""Answers if needle is at least partially overlaps haystack (including boundaries)."""
	hsn = snormalize(haystack)
	nn = snormalize(needle)
	return _soverlaps(hsn, nn)


_normalizationSkippedWarning = " Normalization is skipped."


def _swithin(haystack: SliceRangeT, needle: SliceRangeT) -> bool:
	res = needle.start >= haystack.start and needle.stop <= haystack.stop
	#ic("_swithin", haystack, needle, needle.start >= haystack.start, needle.stop < haystack.stop, res)
	return res


_swithin.__doc__ = swithin.__doc__ + _normalizationSkippedWarning


def _soverlaps(haystack: SliceRangeT, needle: SliceRangeT) -> bool:
	#ic("_soverlaps", haystack, needle, needle.start <= haystack.start < needle.stop, needle.start < haystack.stop < needle.stop)
	return _swithin(haystack, needle) or needle.start <= haystack.start < needle.stop or needle.start < haystack.stop < needle.stop


_soverlaps.__doc__ = soverlaps.__doc__ + _normalizationSkippedWarning


def _teeSliceSequences(sliceSequences: typing.Iterable[SliceRangeSeqT], count: int = 2) -> typing.Iterator[typing.Tuple[itertools._tee, itertools._tee]]:
	for s in sliceSequences:
		if isinstance(s, isInstArg):
			s = (s,)
		yield itertools.tee(s, count)


def teeSliceSequences(sliceSequences: typing.Iterable[SliceRangeSeqT], count: int = 2) -> zip:
	return zip(*(_teeSliceSequences(sliceSequences, count)))


def _integrator(chunkLens: typing.Iterable[int]) -> typing.Iterable[int]:
	cumLen = 0
	for s in chunkLens:
		cumLen += s
		yield cumLen


def _uniq(it: typing.Iterable[typing.Any]) -> typing.Iterable[typing.Any]:
	it = iter(it)
	try:
		prev = next(it)
		yield prev
	except StopIteration:
		return
	for el in it:
		if prev == el:
			continue
		prev = el
		yield el


def _mergeAndDedup(intSeqs: typing.Iterable[typing.Iterable[int]]) -> typing.Iterable[int]:
	return _uniq(sorted(heapq.merge(*intSeqs)))


def _deduplicatedIntegrator(*chunksLens: typing.Iterable[typing.Iterable[int]]) -> typing.Iterable[int]:
	return _mergeAndDedup(map(_integrator, chunksLens))


def ssegments_(slc: SliceRangeT, chunkLens: typing.Iterable[int]) -> SliceRangeSeqT:
	"""Splits the slice into slices of lengths `chunkLen` (which is in `slc.step`s!!!)"""
	return soffset_split_(slc, _deduplicatedIntegrator(chunkLens))  # pylint: disable=undefined-variable


ssegments = _createWrappedWithnewMacroGroup(ssegments_)


def salign_(sliceSequences: typing.Iterable[SliceRangeSeqT]) -> SliceRangeSeqT:
	""""Aligns" seqs of ranges/slices OF THE SAME TOTAL LENGTH, returning ones with additional split points, so that all the sequences have segments of equal lengths between split points with the same indexes. See the test for more insight on what it does."""
	slcsPoints, slcsSplit = teeSliceSequences(sliceSequences, 2)

	splitPoints = tuple(_deduplicatedIntegrator(*(map(_slen, ss) for ss in slcsPoints)))
	for ss in slcsSplit:
		yield soffset_split(ss, splitPoints)  # pylint: disable=undefined-variable
