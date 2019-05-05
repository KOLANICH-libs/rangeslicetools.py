import typing
from abc import ABC, abstractmethod
from collections.abc import Mapping

from .utils import SliceRangeListT, SliceRangeT, salign_, sAny2Type, sjoin_, slen, slice2range, soverlaps, isInstArg, _scollapse, ssegments_
from .utils import sjoin, soffset_split, salign  # pylint: disable=no-name-in-module
from .diff import SDiffAutomata, sdiff, sdist, ssub

__all__ = ("IndexProto", "KeyLeaf", "ValueLeaf", "_RangesTree", "RangesTree", "SliceSequence", "mergeRangesInTreeLookupResult", "FuzzySingleLookupResult", "SingleLookupResult")


# pylint: disable=too-few-public-methods
class IndexProto(Mapping):
	__slots__ = ()

	# in fact a slot
	#@property
	#@abstractmethod
	#def index(self):
	#	raise NotImplementedError()

	def keys(self):
		for n in self:
			yield n.index

	def values(self):
		for n in self:
			yield n.indexee

	def items(self):
		for n in self:
			yield n.index, n.indexee

	def __iter__(self):
		yield self

	def __len__(self):
		return 1


class NodeProto(IndexProto):
	__slots__ = ()

	@property
	@abstractmethod
	def indexee(self):
		raise NotImplementedError()


class ILeaf(NodeProto):

	"""A regular leaf."""
	__slots__ = ("index",)

	def __init__(self, index: SliceRangeT) -> None:
		self.index = index

	def __repr__(self):
		return repr(self.index) + ":[" + repr(self.indexee) + "]"

	def cmpTuple(self) -> typing.Tuple[SliceRangeT, SliceRangeT]:
		return (self.indexee, self.index)

	def __eq__(self, other: "KeyLeaf") -> bool:
		return self.cmpTuple() == other.cmpTuple()

	@property
	def indexee(self) -> SliceRangeListT:
		return self.index

	def __getitem__(self, q: SliceRangeT) -> "LookupResult":
		if soverlaps(self.index, q):
			yield self

	def getPath(self, q, path=()):
		if soverlaps(self.index, q):
			yield SingleLookupResult(self, path)


LookupResult = typing.Iterable["ILeaf"]
LookupPath = typing.Iterable[int]


class SingleLookupResult:
	__slots__ = ("node", "path")

	def __init__(self, node: ILeaf, path: LookupPath):
		self.node = node
		self.path = path

	@property
	def dist(self) -> int:
		return 0

	@property
	def query(self) -> SliceRangeT:
		return self.node.index

	def toTuple(self) -> typing.Tuple[ILeaf, LookupPath, int, typing.Optional[SliceRangeT]]:
		return (self.node, self.path, self.dist, self.query)

	def __eq__(self, other: "SingleLookupResult") -> bool:
		return self.toTuple() == other.toTuple()

	def __repr__(self) -> str:
		return self.__class__.__name__ + "(" + repr(self.node) + ", " + repr(self.path) + ")"


class FuzzySingleLookupResult(SingleLookupResult):
	__slots__ = ("query", "dist",)

	def __init__(self, node: ILeaf, path: LookupPath, dist: int, query: typing.Optional[SliceRangeT]=None):
		super().__init__(node, path)
		self.dist = dist
		self.query = query

	def __repr__(self):
		return super().__repr__()[:-1] + ", " + repr(self.dist) + ", " + repr(self.query) + ")"


class KeyLeaf(ILeaf):
	__slots__ = ()

	@property
	def indexee(self) -> SliceRangeListT:
		return self.index


class ValueLeaf(ILeaf):
	__slots__ = ("_indexee",)

	KEY_LEAF_TYPE = KeyLeaf

	def __init__(self, index: SliceRangeT, indexee: SliceRangeT) -> None:
		super().__init__(index)
		self._indexee = indexee

	@property
	def indexee(self) -> LookupResult:
		return self._indexee

	@indexee.setter
	def indexee(self, v):
		self._indexee = v


def get_lowest_metered(cur, metric) -> LookupResult:
	cur = (cur, None)
	path = ()
	while not isinstance(cur[0], ILeaf):
		cur = min(
			(
				(el, metric(el.index), i) for i, el in enumerate(cur[0].children)
			),
			key=lambda p: p[1]
		)
		path += (cur[2], )
	return FuzzySingleLookupResult(cur[0], path, cur[1])


class _RangesIndexTree(IndexProto):

	"""Allows to store sequences of slices and then query the slices overlapping with the given slice. Returns the whole slices, not their parts."""

	__slots__ = ("_left", "_right", "index")

	INDEX_NODE = ValueLeaf

	def __init__(self) -> None:
		self._left = None
		self._right = None
		self.index = None

	def updateRange(self) -> None:
		if self._left is not None:
			if self._right is not None:
				ress = sjoin((self._left.index, self._right.index))
				self.index = type(ress[0])(ress[0].start, ress[-1].stop, ress[0].step)
			else:
				self.index = self._left.index
		else:
			if self._right is not None:
				self.index = self._right.index
			else:
				raise ValueError("All nodes are empty, cannot compute tree range")

	@property
	def left(self):
		return self._left

	@left.setter
	def left(self, v):
		self._left = v
		self.updateRange()

	@property
	def right(self):
		return self._right

	@right.setter
	def right(self, v):
		self._right = v
		self.updateRange()

	@property
	def children(self):
		return (self.left, self.right)

	@children.setter
	def children(self, v):
		self.left, self.right = v

	def setChild(self, idx: int, newV: IndexProto):
		if idx:
			self.right = newV
		else:
			self.left = newV

	def __iter__(self):
		for el in self.children:
			yield from el

	def __len__(self):
		return sum(len(el) for el in self.children if el)

	def __repr__(self) -> str:
		return (
			repr(self.index) + "[" +
				repr(self.left)
				+ ", " +
				repr(self.right) +
			"]"
		)

	def get_lowest_metered(self, metric) -> LookupResult:
		return get_lowest_metered(self, metric)

	def get_closest(self, q: SliceRangeT):
		intersecting = tuple(self.getPath(q))
		res = list(intersecting)
		fuzzyToMatch = ssub(q, *tuple(el.node.index for el in intersecting))
		for el in fuzzyToMatch:
			def comparator(m):
				return sdist(m, q)
			resPart = self.get_lowest_metered(comparator)
			resPart.query = el
			res.append(resPart)
		return res

	@classmethod
	def build(cls, index: SliceRangeListT, data: typing.Iterable[typing.Any]) -> typing.Union[ILeaf, "RangesTree"]:
		assert len(index) == len(data)
		return cls._build(index=index, ranges=ranges)

	@classmethod
	def _build(cls, index: SliceRangeListT, data: typing.Optional[typing.Iterable[typing.Any]] = None) -> typing.Union[KeyLeaf, "_RangesIndexTree"]:
		#print(__class__.__name__ + ".build", "index=", index, "data=", data)

		if not isinstance(index, isInstArg):
			count = len(index)
			root = cls()
			if count > 1:
				mid = count // 2
				leftIdx, rightIdx = index[:mid], index[mid:]
				if len(leftIdx) == 1:
					leftIdx = leftIdx[0]
				if len(rightIdx) == 1:
					rightIdx = rightIdx[0]

				#print("mid=", mid, "ranges=", ranges)
				if data is not None:
					leftRngs, rightRngs = data[:mid], data[mid:]
					#print("leftRngs=",leftRngs, "rightRngs=",rightRngs)
				else:
					leftRngs, rightRngs = None, None

				root.left = cls._build(index=leftIdx, data=leftRngs)
				root.right = cls._build(index=rightIdx, data=rightRngs)
				return root
		else:
			index = (index,)

		if data:
			data = _scollapse(data)
			return cls.INDEX_NODE(index[0], data)

		return cls.INDEX_NODE.KEY_LEAF_TYPE(index[0])

	def getPath(self, q, path=()):
		if soverlaps(self.index, q):
			for i, ch in enumerate(self.children):
				yield from ch.getPath(q, path + (i,))

	def __getitem__(self, q: SliceRangeT) -> LookupResult:
		for el in self.getPath(q):
			yield el.node

	def getByPath(self, path):
		cur = self
		for el in path:
			cur = cur.children[el]
		return cur

	def getNodesInPath(self, path):
		cur = self
		yield cur
		for el in path:
			cur = cur.children[el]
			yield cur
		yield cur

	def __setitem__(self, k, v) -> None:
		els = self.get_closest(k)
		#print("found", els)
		if len(els) > 1:
			raise NotImplementedError("Value set overlaps multiple leaves: " + repr(els) + ". Not yet implemented, set the leaves individually.")
		elif len(els) == 1:
			for el in els:
				path = tuple(self.getNodesInPath(el.path[:-1]))
				parent = path[-1]
				nod = el.node

				if isinstance(el, FuzzySingleLookupResult):
					# intersecting leaf is not found, found closest leaf. We spawn a new node and create a subtree.

					newParent = self.__class__()
					if k == v:
						newLeaf = KeyLeaf(k)
					else:
						newLeaf = ValueLeaf(k, v)
					if nod.index.start < k.start:
						newParent.children = (nod, newLeaf)
					else:
						newParent.children = (newLeaf, nod)

					parent.setChild(el.path[-1], newParent)

					for pathComp in reversed(path):
						pathComp.updateRange()
				elif isinstance(el, SingleLookupResult):
					if el.node.index == k:
						if isinstance(el.node, ValueLeaf):
							el.node.indexee = v
						elif sinstance(el.node, KeyLeaf):
							replacementLeaf = ValueLeaf(k, v)
							parent.setChild(el.path[-1], replacementLeaf)
						else:
							raise NotImplementedError("Something strange happened: only leaves must be found")
					else:
						if soverlaps(el.index, k):
							others = ssub(k, el.index)
							if others:
								if len(others) > 1:
									other = others[0]
									for othersEl in others:
										newParent = self.__class__()
										if k == v:
											newLeaf = KeyLeaf(k)
										else:
											newLeaf = ValueLeaf(k, v)
										if nod.index.start < other.start:
											nod.index = nod.index.__class__(nod.index.start, other.start)
											newParent.children = (nod, newLeaf)
										else:
											nod.index = nod.index.__class__(other.end, nod.index.end)
											newParent.children = (newLeaf, nod)

										parent.setChild(el.path[-1], newParent)
								elif len(others) == 2:
									newParent1 = self.__class__()
									newParent1.right = newParent2 = self.__class__()

									if kLeft == vLeft:
										newLeafLeft = KeyLeaf(kLeft)
									else:
										newLeafLeft = ValueLeaf(kLeft, vLeft)

									if kRight == vRight:
										newLeafRight = KeyLeaf(kRight)
									else:
										newLeafRight = ValueLeaf(kRight, vRight)
									newLeafLeft.index = others[0]
									newLeafRight.index = others[1]
									middle = nod
									middle.index = k
									middle.data = v

								else:
									raise NotImplementedError("Too many `others`: " + repr(others))
							else:
								el.index = k
								el.indexee = v
		else:
			raise NotImplementedError("Something strange happened: len(els) == " + repr(len(els)))
		#if len(index) == 1:
		#	els[0].indexee =
		#print("aligned", index, ranges, k)


class _RangesTree(_RangesIndexTree):

	"""Allows to store sequences of slices and then query the slices overlapping with the given slice. Returns the whole slices, not their parts."""

	__slots__ = ()

	@classmethod
	def build(cls, index: SliceRangeListT, data: typing.Optional[SliceRangeListT] = None) -> typing.Union[ILeaf, "RangesTree"]:
		if data:
			rangesIsRange = isinstance(data, isInstArg)
			indexIsRange = isinstance(index, isInstArg)
			if (indexIsRange != rangesIsRange) or ((not indexIsRange or not rangesIsRange) and len(data) != len(index)):
				index, data = salign((index, data))
				#print("aligned", index, ranges)

		return cls._build(index=index, data=data)


class RangesTree(_RangesTree):
	__slots__ = ()

	def __getitem__(self, q: typing.Union[SliceRangeT, int]) -> LookupResult:
		if isinstance(q, int):
			q = type(self.range)(q, q + 1)
		return super().__getitem__(q)


class _SliceSequence:
	__slots__ = ("tree",)

	def __init__(self, tree: RangesTree) -> None:
		self.tree = tree

	def __getitem__(self, q: SliceRangeT) -> LookupResult:
		res = list(self.tree[q])
		#print("res", res)
		idxz = []
		valuez = []
		for el in res:
			if isinstance(el, ValueLeaf):
				idx = el.index
				val = el.indexee
			else:
				idx = val = el

			#print("sdiff(", idx, ",", q, ")")
			diff = sdiff(idx, q)
			#from pprint import pprint
			#pprint(diff)

			idx1 = diff[(SDiffAutomata.State.entered, SDiffAutomata.State.entered)]
			idxz.append(idx1)
			idx1Size = slen(idx1)

			leftWasteIdx = (SDiffAutomata.State.entered, SDiffAutomata.State.notEntered)
			if leftWasteIdx in diff:
				#print("leftWaste", diff[leftWasteIdx])
				leftWasteSize = slen(diff[leftWasteIdx])
			else:
				leftWasteSize = 0

			#print(leftWasteSize, ":", leftWasteSize+idx1Size, )
			val = sAny2Type(slice2range(val)[leftWasteSize: leftWasteSize + idx1Size], val.__class__)
			#print("val", val)
			valuez.append(val)

		#print("idx", idx, "idxz", idxz, "valuez", valuez)
		return (ValueLeaf(*p) for p in zip(idxz, valuez))


class SliceSequence(_SliceSequence):

	"""Allows to associate one sequence of slices to another one and then lookup subslices of the second one using subslices in the first one as keys. Useful for ranges rewriting like endianness transformations."""

	__slots__ = ()

	def __init__(self, index: SliceRangeListT, data: typing.Optional[SliceRangeT] = None) -> None:
		#print("SliceSequence.__init__", "data=", data, "index=", index)
		super().__init__(RangesTree.build(index=index, data=data))


def mergeRangesInTreeLookupResult(lookupResults: LookupResult) -> LookupResult:
	idxz, valuez = zip(*((s.index, s.indexee) for s in lookupResults))
	idxz = sjoin(idxz)
	valuez = sjoin_(valuez)
	idxz, valuez = salign_((idxz, valuez))
	return (ValueLeaf(*p) for p in zip(idxz, valuez))
