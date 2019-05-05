import typing
from enum import IntFlag

from .utils import SliceRangeSeqT, SliceRangeT, _isNegative, _sdirect, sjoin_, slen, snormalize

__all__ = ("SDiffAutomata", "sdiff", "sdiffSelectPred_", "sdiffSelect_", "ssub2_", "ssub", "sunion_", "sgap", "sdist")

# pylint: disable=too-few-public-methods


class SDiffAutomata:
	"""
	Stores partiton of the space into the areas where ranges overlap or not.
	"""

	__slots__ = ("state",)

	class State(IntFlag):
		"""
		|<---|
		ee | en | ne

		|--->|
		ne | en | ee
		"""

		notEntered = 0
		entered = 1
		exited = 2

	def __init__(self) -> None:
		self.state = self.__class__.State.notEntered

	def process(self, p: int, isExit: bool) -> None:
		if self.state & self.__class__.State.entered:
			if isExit:
				self.state |= self.__class__.State.exited
			else:
				raise ValueError((p, isExit, self.state))
		else:
			if not isExit:
				self.state |= self.__class__.State.entered
			else:
				raise ValueError((p, isExit, self.state))


sdiffBackDirRemap = {
	SDiffAutomata.State.entered | SDiffAutomata.State.exited: SDiffAutomata.State.notEntered,
	SDiffAutomata.State.entered: SDiffAutomata.State.entered,
	SDiffAutomata.State.notEntered: SDiffAutomata.State.entered | SDiffAutomata.State.exited,
}


IntersectionStateT = typing.Tuple[SDiffAutomata.State, SDiffAutomata.State]


class DiffFSMPoint:
	__slots__ = ("pos", "rangeId", "isEnd")

	def __init__(self, pos, rangeId, isEnd):
		self.pos = pos
		self.rangeId = rangeId
		self.isEnd = isEnd

	@property
	def tuple(self):
		return tuple(getattr(self, k) for k in self.__class__.__slots__)

	def __repr__(self):
		return self.__class__.__name__ + repr(self.tuple)

	def __eq__(self, another):
		return self.tuple == another.tuple

	def __hash__(self):
		return hash(self.tuple)

	def __gt__(self, another):
		return self.pos > another.pos


def _computeEndpointRepresentation(rngs: SliceRangeSeqT):
	points = []
	for i, el in enumerate(rngs):
		points.extend((
			DiffFSMPoint(el.start, i, False),
			DiffFSMPoint(el.stop, i, True),
		))
	points.sort()
	return points


def _endpointsToMatrix(rs, points):
	matrix = {}

	az = [SDiffAutomata() for i in range(len(rs))]

	state = tuple(a.state for a in az)

	for pt in points:
		if state not in matrix:
			matrix[state] = [None, None]
		matrix[state][1] = pt.pos
		az[pt.rangeId].process(pt.pos, pt.isEnd)
		state = tuple(a.state for a in az)
		if state not in matrix:
			matrix[state] = [None, None]
		matrix[state][0] = pt.pos

	del matrix[(SDiffAutomata.State.notEntered, SDiffAutomata.State.notEntered)]
	del matrix[(SDiffAutomata.State.entered | SDiffAutomata.State.exited, SDiffAutomata.State.entered | SDiffAutomata.State.exited)]
	return matrix


def getDirectorRangeIndex(s0: SDiffAutomata.State, s1: SDiffAutomata.State) -> int:
	if (s1 & SDiffAutomata.State.entered and not s1 & SDiffAutomata.State.exited) and not (s0 & SDiffAutomata.State.entered and not s0 & SDiffAutomata.State.exited):
		return 1
	return 0


def _postProcessMatrix(rs, matrix):
	newMatrix = type(matrix)()

	shouldRemapComp = _isNegative(rs)

	for k in matrix:
		el = matrix[k]
		if el[0] == el[1]:
			continue

		directorIdx = getDirectorRangeIndex(*k)
		dR = rs[directorIdx]
		if shouldRemapComp[directorIdx]:
			el = (el[1] - 1, el[0] - 1)

		k = tuple((sdiffBackDirRemap[comp] if shouldRemapComp[i] else comp) for i, comp in enumerate(k))

		newMatrix[k] = dR.__class__(el[0], el[1], dR.step)
	return newMatrix


def sdiff(s0: SliceRangeT, s1: SliceRangeT) -> typing.Dict[IntersectionStateT, SliceRangeT]:
	"""Computes a difference of 2 slices/ranges. More than 2 is not yet implemented, though planned."""
	# pylint: disable=too-many-locals
	rs = (s0, s1)
	canonicalized = snormalize(rs)
	endpoints = _computeEndpointRepresentation(canonicalized)
	return _postProcessMatrix(rs, _endpointsToMatrix(rs, endpoints))


def sdiffSelectPred_(s1: SliceRangeT, s2: SliceRangeT, pred) -> SliceRangeSeqT:
	"""Computes differences and takes states."""
	res = sdiff(s1, s2)
	for k, v in res.items():
		if pred(k):
			yield v


def sdiffSelect_(s1: SliceRangeT, s2: SliceRangeT, keys: list, neg: bool = False) -> SliceRangeSeqT:
	"""Computes differences and takes states."""
	res = sdiff(s1, s2)
	for k in keys:
		v = res.get(k, None)
		if (v is not None) != neg:
			yield res[k]


def ssub2_(s1: SliceRangeT, s2: SliceRangeT) -> SliceRangeSeqT:
	"""Subtracts 2 ranges"""
	S = SDiffAutomata.State
	return sdiffSelect_(s1, s2, [(S.entered, S.notEntered), (S.entered, S.entered | S.exited)])


def ssub(s1: SliceRangeT, *rest: typing.Iterable[SliceRangeT]) -> SliceRangeSeqT:
	"""Subtracts n >= 1 ranges"""
	res = (s1,)
	for i, el2 in enumerate(rest):
		res1 = ()
		for el1 in res:
			res1 += ssub2(el1, el2)  # pylint:disable=undefined-variable
		res = res1
	return res


def sunion_(s1: SliceRangeT, s2: SliceRangeT) -> SliceRangeSeqT:
	"""Unions 2 ranges.
	See also `shull`, which is faster, but dumber.
	"""
	S = SDiffAutomata.State

	def pred(k):
		a, b = k
		return (a & S.entered and not a & S.exited) or (b & S.entered and not b & S.exited)

	toJoin = sorted(sdiffSelectPred_(s1, s2, pred), key=lambda e: snormalize(e).start)
	return sjoin_(toJoin)


def sgap(s1: SliceRangeT, s2: SliceRangeT) -> typing.Optional[SliceRangeT]:
	"""Returns a gap between 2 ranges"""
	S = SDiffAutomata.State

	def pred(k):
		a, b = k
		return (a & S.exited or not a & S.entered) and (b & S.exited or not b & S.entered)

	try:
		return next(sdiffSelectPred_(s1, s2, pred))
	except StopIteration:
		return None


def sdist(s1: SliceRangeT, s2: SliceRangeT) -> int:
	"""Returns length of a gap between 2 ranges"""
	gap = sgap(s1, s2)
	if gap:
		return slen(gap)

	return 0
