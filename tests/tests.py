#!/usr/bin/env python3
import typing
import os, sys
import unittest
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

from rangeslicetools import *
from rangeslicetools.utils import _getStepForComputation, isInstArg


def constructNestedRangeSliceSeq(ctor, seq):
	if seq:
		if isinstance(seq[0], int):
			return ctor(*seq)
		else:
			return seq.__class__(constructNestedRangeSliceSeq(ctor, el) for el in seq)
	return seq


cnss = constructNestedRangeSliceSeq

#@unittest.skip
class TestTestUtils(unittest.TestCase):
	def test_constructNestedRangeSliceSeq(self):
		pairs = {
			(1, 2): range(1, 2),
			((1, 2), (3, 4)): (range(1, 2), range(3, 4)),
			(((1, 2),), ((3, 4),(5, 6))): ((range(1, 2),), (range(3, 4), range(5, 6)))
		}
		for challenge, response in pairs.items():
			with self.subTest(challenge=challenge, response=response):
				self.assertEqual(constructNestedRangeSliceSeq(range, challenge), response)


#@unittest.skip
class Tests(unittest.TestCase):
	def test_getStepForComputation(self) -> None:
		pairs = {
			(0, 16,): 1,
			(0, 16, 1): 1,
			(0, 16, 2): 2,
			(15, -1, -1): -1,
			(15, -1, -2): -2,
		}
		for ctor in isInstArg:
			for challenge, response in pairs.items():
				challenge = ctor(*challenge)

				with self.subTest(challenge=challenge, response=response):
					self.assertEqual(_getStepForComputation(challenge), response)

	def _test_any2any(self, ctorSrc: typing.Union[typing.Type[slice], typing.Type[range]], ctorDst: typing.Union[typing.Type[slice], typing.Type[range]], convertor: typing.Callable) -> None:
		testRanges = [
			(0, 16, 1),
			(17, -1, -2),
		]

		for r in testRanges:
			src = ctorSrc(*r)
			exp = ctorDst(*r)
			with self.subTest(src=src, ctorSrc=ctorSrc, convertor=convertor):
				self.assertEqual(convertor(src), exp)

	def test_slice2range(self) -> None:
		self._test_any2any(slice, range, slice2range)
		self._test_any2any(range, range, slice2range)

	def test_range2slice(self) -> None:
		self._test_any2any(range, slice, range2slice)
		self._test_any2any(slice, slice, range2slice)

	def test_slen(self) -> None:
		testRanges = [
			(0, 16, 1),
			(17, -1, -2),
			(15, -1, -1),
			(0, 16),
			(17, -1, -1),
			(15, -1, -2),
		]

		for ctor in isInstArg:
			sumLen = 0

			for r in testRanges:
				r = cnss(ctor, r)
				with self.subTest(r=r):
					l = len(list(slice2range(r)))
					self.assertEqual(slen(r), l)
					sumLen += l

			with self.subTest(r=testRanges):
				self.assertEqual(slen(cnss(ctor, testRanges)), sumLen)

	def test_sPointIn(self) -> None:
		pairs = {
			((0, 16, 1), 15): True,
			((0, 16, 1), 18): False,
			((0, 16, 1), 16): False,
			((0, 16, 1), 0): True,

			((15, -1, -1), 15): True,
			((15, -1, -1), 18): False,
			((15, -1, -1), 16): False,
			((15, -1, -1), 0): True,
		}

		for ctor in isInstArg:
			sumLen = 0

			for challenge, response in pairs.items():
				r, p = challenge
				r = cnss(ctor, r)
				with self.subTest(challenge=challenge, response=response):
					self.assertEqual(sPointIn(r, p), response)

	def test_svec(self) -> None:
		pairs = (
			((0, 16, 1), 16),
			((14, -2, -2), -8)
		)
		for ctor in isInstArg:
			for s, orientedLen in pairs:
				s = cnss(ctor, s)
				with self.subTest(s=s):
					self.assertEqual(svec(s), orientedLen)

	def test_srev(self) -> None:
		pairs = (
			((0, 16, 1), (15, -1, -1)),
			((0, 16, 2), (14, -2, -2))
		)
		for ctor in isInstArg:
			for s, rev in pairs:
				s = cnss(ctor, s)
				rev = cnss(ctor, rev)
				resS = srev(rev)

				with self.subTest(s=s, rev=rev, resS=resS):
					self.assertEqual(list(slice2range(resS)), list(slice2range(s)))
					self.assertEqual(resS, s)

				try:
					resRev = srev(s)
				except BaseException:
					continue
				with self.subTest(s=s, rev=rev, resRev=resRev):
					self.assertEqual(list(slice2range(resRev)), list(slice2range(rev)))
					self.assertEqual(resRev, rev)

	def test_sdir(self) -> None:
		pairs = (
			((0, 16, 1), 1),
			((14, -2, -2), -1),
			((0, 16), 1),
			((14, -2), -1),
		)
		for ctor in isInstArg:
			for s, exp in pairs:
				s = cnss(ctor, s)

				with self.subTest(s=s):
					self.assertEqual(sdir(s), exp)

	def test_sdirect(self) -> None:
		pairs = {
			((0, 16, 1), (1, 2, 1)): (1, 2, 1),
			((0, 16, 1), (1, 0, -1)): (1, 2, 1),
			((16, 0, -1), (1, 2, 1)): (1, 0, -1),
			((16, 0, -1), (1, 0, -1)): (1, 0, -1),
		}
		for ctor in isInstArg:
			for ctor2 in isInstArg:
				for (donor, acceptor), expected in pairs.items():
					donor = cnss(ctor, donor)
					acceptor = ctor2(*acceptor)
					expected = ctor2(*expected)

					with self.subTest(donor=donor, acceptor=acceptor):
						self.assertEqual(sdirect(donor, acceptor), expected)

	def test_snormalize(self) -> None:
		testRanges = [
			(0, 16, 1),
			(0, 16, 2)
		]

		for ctor in isInstArg:
			r = ctor(0, 16)
			with self.subTest(r=r):
				self.assertEqual(snormalize(r), ctor(0, 16, 1))

		for ctor in isInstArg:
			for r in testRanges:
				r = cnss(ctor, r)
				with self.subTest(r=r):
					self.assertEqual(snormalize(r), r)

				try:
					rev = srev(r)
				except AssertionError:
					continue
				with self.subTest(rev=rev):
					self.assertEqual(snormalize(rev), r)


@unittest.skip
class RelationTests(unittest.TestCase):
	def _testRelation(self, pairs, func: typing.Callable) -> None:
		for ctor in isInstArg:
			for initialRanges, expectedResult in pairs.items():
				initialRanges = cnss(ctor, initialRanges)

				with self.subTest(initialRanges=initialRanges):
					self.assertEqual(func(*initialRanges), expectedResult)

	def test_swithin(self) -> None:
		pairs = {
			((0, 10), (1, 5)): True,
			((0, 10), (-5, 3)): False,
			((0, 10), (7, 15)): False,

			((9, -1, -1), (1, 5)): True,
			((9, -1, -1), (-5, 3)): False,
			((9, -1, -1), (7, 15)): False,

			((0, 10), (4, 0, -1)): True,
			((0, 10), (2, -6, -1)): False,
			((0, 10), (14, 6, -1)): False,

			((9, -1, -1), (4, 0, -1)): True,
			((9, -1, -1), (2, -6, -1)): False,
			((9, -1, -1), (14, 6, -1)): False,


			((0, 10), (11, 15)): False,
			((0, 10), (-15, -11)): False,

			((9, -1, -1), (11, 15)): False,
			((9, -1, -1), (-15, -11)): False,

			((0, 10), (14, 10, -1)): False,
			((0, 10), (-12, -16, -1)): False,

			((9, -1, -1), (14, 10, -1)): False,
			((9, -1, -1), (-12, -16, -1)): False,
		}
		self._testRelation(pairs, swithin)

	def test_soverlaps(self) -> None:
		pairs = {
			((0, 10), (1, 5)): True,
			((0, 10), (-5, 3)): True,
			((0, 10), (7, 15)): True,

			((9, -1, -1), (1, 5)): True,
			((9, -1, -1), (-5, 3)): True,
			((9, -1, -1), (7, 15)): True,

			((0, 10), (4, 0, -1)): True,
			((0, 10), (2, -6, -1)): True,
			((0, 10), (14, 6, -1)): True,

			((9, -1, -1), (4, 0, -1)): True,
			((9, -1, -1), (2, -6, -1)): True,
			((9, -1, -1), (14, 6, -1)): True,


			((0, 10), (11, 15)): False,
			((0, 10), (-15, -11)): False,

			((9, -1, -1), (11, 15)): False,
			((9, -1, -1), (-15, -11)): False,

			((0, 10), (14, 10, -1)): False,
			((0, 10), (-12, -16, -1)): False,

			((9, -1, -1), (14, 10, -1)): False,
			((9, -1, -1), (-12, -16, -1)): False,


			((0, 4, 1), (2, 4, 1)): True,
			((1, 2), (2, 4, 1)): False,
		}
		self._testRelation(pairs, soverlaps)


#@unittest.skip
class SplitMergeTests(unittest.TestCase):
	def _testSplit(self, pairs, func: typing.Callable) -> None:
		for ctor in isInstArg:
			for argsTuple, expectedResult in pairs.items():
				initialRange = argsTuple[0]
				otherArgs = argsTuple[1:]
				initialRange = cnss(ctor, initialRange)
				expectedResult = cnss(ctor, expectedResult)

				with self.subTest(initialRange=initialRange, otherArgs=otherArgs):
					self.assertEqual(func(initialRange, *otherArgs), expectedResult)

	def test_ssplit(self) -> None:
		pairs = {
			((0, 8), (2, 3, 7)): ((0, 2), (2, 3), (3, 7), (7, 8)),
			((0, 8), (2, 8)): ((0, 2), (2, 8)),
			((7, -1, -1), 2): ((7, 2, -1), (2, -1, -1)),
			#(((0, 4), (4, 8), (8, 12), (12, 16)), 8): (((0, 4), (4, 8)), ((8, 12), (12, 16))),
		}
		self._testSplit(pairs, ssplit)

	def test_ssplit_1(self) -> None:
		pairs = {
			((0, 8), (2, 3, 7)): ((0, 2), (2, 3), (3, 7), (7, 8)),
			((0, 8), (2, 8)): ((0, 2), (2, 8), (8, 8)),
			((7, -1, -1), 2): ((7, 2, -1), (2, -1, -1)),
		}
		self._testSplit(pairs, ssplit_1)

	def test_schunks(self) -> None:
		pairs = {
			((0, 8), 8): ((0, 8),),
			((0, 8), 9): ((0, 8),),
			((0, 8), 2): ((0, 2), (2, 4), (4, 6), (6, 8)),
			((1, 9, 1), 4): ((1, 5, 1), (5, 9, 1)),
			((1, 9, 4), 8): ((1, 9, 4),),
			((7, -1, -1), 2): ((7, 5, -1), (5, 3, -1), (3, 1, -1), (1, -1, -1)),
		}
		self._testSplit(pairs, schunks)

	def test_ssegments(self) -> None:
		pairs = {
			((0, 8), (2, 3, 3)): ((0, 2), (2, 5), (5, 8)),
			((0, 8), (2, 3, 7)): ((0, 2), (2, 5), (5, 8)),
			((0, 8), (2, 6)): ((0, 2), (2, 8)),
			((7, -1, -1), (2,)): ((7, 5, -1), (5, -1, -1)),
		}
		self._testSplit(pairs, ssegments)

	def test_soffset_split(self) -> None:
		pairs = {
			((0, 8, 1), (-3, 9, 10)): ((0, 8, 1),),
			((0, 8, 1), (2, 3, 7)): ((0, 2, 1), (2, 3, 1), (3, 7, 1), (7, 8, 1)),
			((1, 8, 1), (2, 6)): ((1, 3, 1), (3, 7, 1), (7, 8, 1)),
			((1, 8), (2, 7)): ((1, 3), (3, 8)),
			((7, -1, -1), 2): ((7, 5, -1), (5, -1, -1)),
			((7, -1, -1), ()): ((7, -1, -1),),
			(((0, 4), (4, 8), (8, 12), (12, 16)), 8): (((0, 4), (4, 8)), ((8, 12), (12, 16))),
		}
		self._testSplit(pairs, soffset_split)

	def test_shull(self) -> None:
		pairs = {
			((0, 5), (5, 8)): (0, 8),
			((8, 7, -1), (7, -1, -1)): (8, -1, -1),
			((5, 8), (0, 5)): (0, 8),
			((8, 7, -1), (9, 8, -1)): (9, 7, -1),
			((0, 7), (5, 8)): (0, 8),
			((5, 8), (0, 7),): (0, 8),
			((8, 6, -1), (7, -1, -1)): (8, -1, -1),
			((7, -1, -1), (8, 6, -1)): (8, -1, -1),
			((0, 9, 1), (10, 12, 1)): (0, 12, 1),
			((10, 12, 1), (0, 9, 1),): (0, 12, 1),
			((8, 7, -1), (4, -1, -1)): (8, -1, -1),
			((9, 10, 1), (0, 12, 1)): (0, 12, 1),
			((10, 12, 1), (0, 9, 1),): (0, 12, 1),
		}

		for ctor in (range,):
			for initialRanges, expectedResult in pairs.items():
				initialRanges = cnss(ctor, initialRanges)
				expectedResult = cnss(ctor, expectedResult)

				with self.subTest(initialRanges=initialRanges):
					self.assertEqual(shull(initialRanges), expectedResult)

	def test_sjoin(self) -> None:
		pairs = {
			#[~~~5~>]
			#        [~~3~>]
			#0.......5.....8
			#[~~~5~~~~~~3~>]
			((0, 5), (5, 8)): ((0, 8),),

			#       [<~1~~]
			#[<~8~~~]
			#0......8.....9
			#[<~8~~~~~~1~~]
			((8, 7, -1), (7, -1, -1)): ((8, -1, -1),),

			#       [~~3~>]
			#[~~~5~>]
			#0......6.....8
			((5, 8), (0, 5)): ((5, 8), (0, 5),),

			#[<~1~~~]
			#.......[<~1~~]
			#8......9.....10
			((8, 7, -1), (9, 8, -1)): ((8, 7, -1), (9, 8, -1),),

			#[~~~7~~~~>]
			#        [~~2~~>]
			#0.......5.7....8
			#[~~~~~~~8~~~~~>]
			((0, 7), (5, 8)): ((0, 8),),

			#        [~~2~~>]
			#[~~~7~~~~>]
			#0.......5.7....8
			((5, 8), (0, 7),): ((5, 8), (0, 7),),

			#............[<~~~2~~~~~]
			#[<~~~~~8~~~~~~]
			#.0.....7.......8.......9
			((8, 6, -1), (7, -1, -1)): ((8, -1, -1),), #!!!!


			((0, 5), (5, 10)): ((0, 10),),
			((0, 4), (4, 8), (8, 10)): ((0, 10),),
			((0, 5), (5, 10)): ((0, 10),),
			((0, 4), (4, 8), (8, 10)): ((0, 10),),

			#[<~~~~~8~~~~~~~]
			#...........[<~~~2~~~~~]
			#0.....7.......8.......9
			((7, -1, -1), (8, 6, -1)): ((7, -1, -1), (8, 6, -1)),

			
			#[~~~9~>]....1...[~~2~>]
			#0......9........10....12
			((0, 9, 1), (10, 12, 1)): ((0, 9, 1), (10, 12, 1)),
			((10, 12, 1), (0, 9, 1),): ((10, 12, 1), (0, 9, 1),),

			#[<~5~~~]...3...[<~1~~]
			#0......5.......8.....9
			((8, 7, -1), (4, -1, -1)): ((8, 7, -1), (4, -1, -1)),

			#.......[~~~1~~~>]......
			#[~~~~~~~12~~~~~~~~~~~~>]
			#0......9........10....12
			#[~~~~~~~12~~~~~~~~~~~~>]
			((9, 10, 1), (0, 12, 1)): ((0, 12, 1),),
			((10, 12, 1), (0, 9, 1),): ((10, 12, 1), (0, 9, 1),),

			((0, 8, 1), (8, 9, 1), (9, 10, 1)): ((0, 10, 1),),
			((0, 8), (8, 9), (9, 10)): ((0, 10),),
			((0, 8, 1), (8, 9, 1), (10, 12, 1), (12, 14, 1)): ((0, 9, 1), (10, 14, 1)),
			((0, 8, 2), (8, 10, 2), (11, 13, 2), (13, 15, 2)): ((0, 10, 2), (11, 15, 2)),
			((0, 8), (8, 9), (10, 12), (12, 14)): ((0, 9), (10, 14)),
			((7, -1, -1), (8, 7, -1)): ((7, -1, -1), (8, 7, -1),),
			((9, 8, -1), (8, 7, -1), (7, -1, -1)): ((9, -1, -1),),
			((7, -1, -1), (8, 7, -1), (9, 8, -1)): ((7, -1, -1), (8, 7, -1), (9, 8, -1),),
			(): (),
		}

		for ctor in (range,):
			for initialRanges, expectedResult in pairs.items():
				initialRanges = cnss(ctor, initialRanges)
				expectedResult = cnss(ctor, expectedResult)

				with self.subTest(initialRanges=initialRanges):
					self.assertEqual(sjoin(initialRanges), expectedResult)


	def test_salign(self) -> None:
		testMatrix = {
			(
				((9, 8, -1), (8, 7, -1), (7, -1, -1)),
				((19, 15, -1), (15, 13, -1), (13, 9, -1)),
			):(
				(( 9,  8, -1), ( 8,  7, -1), ( 7,  5, -1), ( 5,  3, -1), ( 3, -1, -1)),
				((19, 18, -1), (18, 17, -1), (17, 15, -1), (15, 13, -1), (13,  9, -1)),
			),
			# my
			(
				((15, -1, -1),),
				((7, -1, -1), (15, 7, -1)),
			):(
				((15, 7, -1), (7, -1, -1)),
				((7, -1, -1), (15, 7, -1)),
			),
			# my
			(
				((7, -1, -1), (15, 7, -1)),
				(15, -1, -1),
			):(
				((7, -1, -1), (15, 7, -1)),
				((15, 7, -1), (7, -1, -1)),
			),
		}
		for ctor in isInstArg:
			for chall, expectedRes in testMatrix.items():

				def transformChall():
					for rsIt in chall:
						yield cnss(ctor, rsIt)

				chall = tuple(transformChall())
				expectedRes = cnss(ctor, expectedRes)
				with self.subTest(chall=chall):
					res = salign(chall)
					self.assertEqual(res, expectedRes)


#@unittest.skip
class DiffTests(unittest.TestCase):
	def test_sdiff(self) -> None:
		S = SDiffAutomata.State

		pairs = {
			#  0        5      7       10
			# |%------->%
			# %        |%------%------>%%
			# |%-en,ne-}%----ee,en----}%%
			((0, 5, 1), (5, 10, 1)): {
				(S.entered, S.notEntered): (0, 5, 1),
				(S.entered | S.exited, S.entered): (5, 10, 1),
			},
			# 0        5        7        10
			#|%--------%------->%
			#         |%--------%------->%%
			#|%-en,ne-}%en,en--}%-ee,en-}%%
			((0, 7, 1), (5, 10, 1)): {
				(S.entered, S.notEntered): (0, 5, 1),
				(S.entered, S.entered): (5, 7, 1),
				(S.entered | S.exited, S.entered): (7, 10, 1),
			},
			# 0        5       7       10
			#|%----------------------->%%
			#         |%------>%
			#|%-en,ne-}%en,en-}%en,ee-}%%
			((0, 10, 1), (5, 7, 1)): {
				(S.entered, S.notEntered): (0, 5, 1),
				(S.entered, S.entered): (5, 7, 1),
				(S.entered, S.entered | S.exited): (7, 10, 1),
			},

			# -1         7        15
			#            $<-------$$|
			# $$<-----------------$$|
			#   {-ex,en--${-en,en-$$|
			((15, 7, -1), (15, -1, -1)): {
				(S.exited | S.entered, S.entered): (7, -1, -1),
				(S.entered, S.entered): (15, 7, -1),
			},

			# -1        1        2       7
			# $$<-------$--------$-------$|
			#           $<-------$|
			# $${-en,ee-${-en,en-${-en,ne$|
			((7, -1, -1), (2, 1, -1)): {
				(S.entered, S.notEntered): (7, 2, -1),
				(S.entered, S.entered): (2, 1, -1),
				(S.entered, S.exited | S.entered): (1, -1, -1)
			},

            # mixed direction

			#  0       4 5      7     9
			# |%-------$>%
			#  %       $<%------------$|
			# |%-en,ee-$X%----ee,en---$|
			((0, 5, 1), (9, 4, -1)): {
				(S.entered, S.entered | S.exited): (0, 5, 1),
				(S.entered | S.exited, S.entered): (9, 4, -1),
			},
			#-1        4 5      7       10
			#$$<-------$|%
			#$$        $|%------%------>%%
			#$$(-en,ne-$|%----ne,en----}%%
			((4, -1, -1), (5, 10, 1)): {
				(S.entered, S.notEntered): (4, -1, -1),
				(S.notEntered, S.entered): (5, 10, 1),
			},

			# 0       4          7      9
			#|%-------$--------->%
			#         $<---------%------$|
			#|%-en,ee-$}-en,en--{%-ee,en$|
			((0, 7, 1), (9, 4, -1)): {
				(S.entered, S.entered | S.exited): (0, 5, 1),
				(S.entered, S.entered): (5, 7, 1),
				(S.entered | S.exited, S.entered): (9, 6, -1),
			},
			#-1         5      6         10
			#$$<--------%------$|
			#$$        |%------$-------->%%
			#$${-en,ne-{%en,en-$|-ne,en-}%%
			((6, -1, -1), (5, 10, 1)): {
				(S.entered, S.notEntered): (4, -1, -1),
				(S.entered, S.entered): (6, 4, -1),
				(S.notEntered, S.entered): (7, 10, 1),
			},

			# 0       4       6         10
			#|%-------$-------$-------->%%
			#         $<------$|
			#|%-en,ee-$X-en,en$|-en,ne-}%%
			((0, 10, 1), (6, 4, -1)): {
				(S.entered, S.entered | S.exited): (0, 5, 1),
				(S.entered, S.entered): (5, 7, 1),
				(S.entered, S.notEntered): (7, 10, 1),
			},
			#-1         5       7     9
			#$$<--------%-------%-----$|
			#$$        |%------>%
			#$${-en,ne-{%en,en-{%en,ee$|
			((9, -1, -1), (5, 7, 1)): {
				(S.entered, S.notEntered): (4, -1, -1),
				(S.entered, S.entered): (6, 4, -1),
				(S.entered, S.entered | S.exited): (9, 6, -1),
			},
		}

		for ctor in isInstArg:
			for chall, resp in pairs.items():
				chall = cnss(ctor, chall)
				with self.subTest(chall=chall):
					self.assertEqual(
						sdiff(*chall),
						{
							k: cnss(ctor, v) for k, v in resp.items()
						}
					)

	def test_sunion(self) -> None:
		pairs = {
			((0, 5, 1), (5, 10, 1)): ((0, 10, 1),),
			((0, 8, 1), (4, 10, 1)): ((0, 10, 1),),
			((5, 10, 1), (0, 5, 1)): ((0, 10, 1),),
			((4, 10, 1), (0, 8, 1)): ((0, 10, 1),),

			##failed
			#((4, -1, -1), (5, 10, 1)): ((4, -1, -1), (9, 4, -1)),
			#((0, 8, 1), (9, 3, -1)): ((0, 10, 1),),
			#((9, 3, -1), (0, 8, 1)): ((9, -1, -1),),
			#((5, 10, 1), (4, -1, -1)): ((0, 10, 1),),
			#((4, -1, -1), (9, 4, -1)): ((9, -1, -1),),
			#((9, 3, -1), (7, -1, -1)): ((9, -1, -1),),
			#((7, -1, -1), (9, 3, -1)): ((9, -1, -1),),
			#((9, 4, -1), (4, -1, -1)): ((9, -1, -1),),
		}

		for ctor in isInstArg:
			for chall, resp in pairs.items():
				chall = cnss(ctor, chall)
				with self.subTest(chall=chall):
					resp = cnss(ctor, resp)
					self.assertEqual(
						sunion(*chall),
						resp
					)

	def test_ssub(self) -> None:
		pairs = {
			((0, 5, 1),): ((0, 5, 1),),
			((0, 7, 1), (5, 10, 1)): ((0, 5, 1),),
			((0, 10, 1), (5, 7, 1)): ((0, 5, 1), (7, 10, 1)),
			((0, 10, 1), (5, 7, 1), (1, 8, 1)): ((0, 1, 1), (8, 10, 1)),
			((15, 7, -1), (15, -1, -1)): (),
			((7, -1, -1), (2, 1, -1)): ((7, 2, -1), (1, -1, -1)),
			((0, 5, 1), (7, 10, 1)): ((0, 5, 1),),
			((7, 10, 1), (0, 5, 1)): ((7, 10, 1),),
			((9, 6, -1), (4, -1, -1)): ((9, 6, -1),),
			((4, -1, -1), (9, 6, -1)): ((4, -1, -1),)
		}

		for ctor in isInstArg:
			for chall, resp in pairs.items():
				chall = cnss(ctor, chall)
				with self.subTest(chall=chall):
					resp = cnss(ctor, resp)
					self.assertEqual(
						resp,
						ssub(*chall),
					)

	def test_sgap(self) -> None:
		pairs = {
			((0, 5, 1), (5, 10, 1)): None,
			((0, 7, 1), (5, 10, 1)): None,
			((0, 10, 1), (5, 7, 1)): None,
			((15, 7, -1), (15, -1, -1)): None,
			((7, -1, -1), (2, 1, -1)): None,
			((0, 5, 1), (7, 10, 1)): (5, 7, 1),
			((7, 10, 1), (0, 5, 1)): (5, 7, 1),
			((9, 6, -1), (4, -1, -1)): (6, 4, -1),
			((4, -1, -1), (9, 6, -1)): (6, 4, -1),
		}

		for ctor in isInstArg:
			for chall, resp in pairs.items():
				chall = cnss(ctor, chall)
				with self.subTest(chall=chall):
					if resp is not None:
						resp = cnss(ctor, resp)
					self.assertEqual(
						resp,
						sgap(*chall),
					)

	def test_sdist(self) -> None:
		pairs = {
			((0, 5, 1), (5, 10, 1)): 0,
			((0, 7, 1), (5, 10, 1)): 0,
			((0, 10, 1), (5, 7, 1)): 0,
			((15, 7, -1), (15, -1, -1)): 0,
			((7, -1, -1), (2, 1, -1)): 0,
			((0, 5, 1), (7, 10, 1)): 2,
			((7, 10, 1), (0, 5, 1)): 2,
			((9, 6, -1), (4, -1, -1)): 2,
			((4, -1, -1), (9, 6, -1)): 2,
		}

		for ctor in isInstArg:
			for chall, resp in pairs.items():
				chall = cnss(ctor, chall)
				with self.subTest(chall=chall):
					self.assertEqual(
						sdist(*chall),
						resp
					)


class IndexTestsProto(unittest.TestCase):
	indexerCtor = None

	@staticmethod
	def _genResItem(el, ctorSrc: SliceRangeT):
		if isinstance(el, ValueLeaf):
			return ValueLeaf(index=ctorSrc(*el.index), indexee=ctorSrc(*el.indexee))
		else:
			return KeyLeaf(index=ctorSrc(*el))

	@classmethod
	def _genResult(cls, res: typing.Any, ctorSrc: SliceRangeT) -> typing.Iterator[KeyLeaf]:
		for el in res:
			yield cls._genResItem(el, ctorSrc)

	def _testIndex(self, index: typing.Optional[typing.Tuple[int, int, int]], testMatrix: typing.Dict[typing.Tuple[int, int, int], typing.Any], src: typing.List[typing.Tuple[int, int, int]] = None) -> None:
		for ctorSrc in isInstArg:
			t = self.__class__.indexerCtor(index=cnss(ctorSrc, index), data=(cnss(ctorSrc, src) if src is not None else None))
			#ic("t", t)

			for ctorQuery in isInstArg:
				for q, expectedRes in testMatrix.items():
					q = ctorQuery(*q)
					#ic("q", q)
					with self.subTest(q=q):
						expectedRes = tuple(self.__class__._genResult(expectedRes, ctorSrc))
						res = tuple(t[q])
						#ic("expectedRes", expectedRes)
						#ic("res", res)
						self.assertEqual(res, expectedRes)


#@unittest.skip
class TreeTests(IndexTestsProto):
	indexerCtor = RangesTree.build

	def testTreeTrivial(self) -> None:
		rng = (7, -1, -1)
		src = [rng]
		index = rng

		def genTests():
			for i in range(*rng):
				for j in range(i, *rng[1:]):
					q = (i, j, rng[2])
					yield (q, (ValueLeaf(rng, rng),))

		matrix = dict(genTests())
		self._testIndex(index, matrix, src)

	def testsTreeDumb(self) -> None:
		index = ((0, 4, 1), (4, 8, 1), (8, 12, 1), (12, 16, 1))
		matrix = {
			(0, 8, 1): ((0, 4, 1), (4, 8, 1)),
			(8, 16, 1): ((8, 12, 1), (12, 16, 1)),

			(4, 12, 1): ((4, 8, 1), (8, 12, 1)),
			(4, 8, 1): ((4, 8, 1),),

			(1, 15, 1): ((0, 4, 1), (4, 8, 1), (8, 12, 1), (12, 16, 1)),
			(5, 15, 1): ((4, 8, 1), (8, 12, 1), (12, 16, 1)),
			(9, 15, 1): ((8, 12, 1), (12, 16, 1)),

			(12, 15, 1): ((12, 16, 1),),
			(3, 4, 1): ((0, 4, 1),),
			(0, 1, 1): ((0, 4, 1),),
			(7, 9, 1): ((4, 8, 1), (8, 12, 1)),
		}
		self._testIndex(index, matrix)

	def testsTreeDisjoint(self) -> None:
		index = ((0, 9, 1), (14, 32, 1), (127, 255, 1))
		matrix = {
			(-1, 0, 1): (),
			(0, 1, 1): ((0, 9, 1),),
			(9, 10, 1): (),
			(-1, 10, 1): ((0, 9, 1),),
			(-1, 5, 1): ((0, 9, 1),),
			(5, 10, 1): ((0, 9, 1),),

			(13, 14, 1): (),
			(14, 15, 1): ((14, 32, 1),),
			(32, 33, 1): (),
			(13, 33, 1): ((14, 32, 1),),
			(13, 23, 1): ((14, 32, 1),),
			(23, 33, 1): ((14, 32, 1),),

			(126, 127, 1): (),
			(127, 128, 1): ((127, 255, 1),),
			(32, 256, 1): (),
			(126, 256, 1): ((127, 255, 1),),
			(126, 165, 1): ((127, 255, 1),),
			(32, 256, 1): ((127, 255, 1),),
		}
		self._testIndex(index, matrix)

	def testsTreeDumbNegatives(self) -> None:
		index = ((0, 4, 1), (4, 8, 1), (8, 12, 1), (12, 16, 1))

		matrix = {
			(7, -1, -1): ((0, 4, 1), (4, 8, 1)),
			(15, 7, -1): ((8, 12, 1), (12, 16, 1)),

			(11, 3, -1): ((4, 8, 1), (8, 12, 1)),
			(7, 3, -1): ((4, 8, 1),),

			(14, 0, -1): ((0, 4, 1), (4, 8, 1), (8, 12, 1), (12, 16, 1)),
			(14, 4, -1): ((4, 8, 1), (8, 12, 1), (12, 16, 1)),
			(14, 8, -1): ((8, 12, 1), (12, 16, 1)),

			(14, 11, -1): ((12, 16, 1),),
			(3, 2, -1): ((0, 4, 1),),
			(0, -1, -1): ((0, 4, 1),),
			(8, 6, -1): ((4, 8, 1), (8, 12, 1)),
		}
		self._testIndex(index, matrix)

	def testsTreeReversedDumb(self) -> None:
		index = (
			(3, 2, -1),
			(2, 1, -1),
			(1, 0, -1),
			(0, -1, -1),
		)
		#ic(t)

		matrix = {
			(0, 2, 1): ((1, 0, -1), (0, -1, -1)),
			(2, 4, 1): ((3, 2, -1), (2, 1, -1)),

			(1, 3, 1): ((2, 1, -1), (1, 0, -1)),
			(1, 2, 1): ((1, 0, -1),),
		}
		self._testIndex(index, matrix)

	def testsTreeIndexedPositiveDumb(self) -> None:
		src = ((0, 4, 1), (4, 8, 1), (8, 12, 1), (12, 16, 1))
		index = (-16, 0, 1)

		matrix = {
			(-16, -8, 1): (ValueLeaf((-16, -12, 1), (0, 4, 1)), ValueLeaf((-12, -8, 1), (4, 8, 1))),
			(-8, 0, 1): (ValueLeaf((-8, -4, 1), (8, 12, 1)), ValueLeaf((-4, 0, 1), (12, 16, 1))),

			(-12, -4, 1): (ValueLeaf((-12, -8, 1), (4, 8, 1)), ValueLeaf((-8, -4, 1), (8, 12, 1))),
			(-12, -8, 1): (ValueLeaf((-12, -8, 1), (4, 8, 1)),),

			(-15, -1, 1): (ValueLeaf((-16, -12, 1), (0, 4, 1)), ValueLeaf((-12, -8, 1), (4, 8, 1)), ValueLeaf((-8, -4, 1), (8, 12, 1)), ValueLeaf((-4, 0, 1), (12, 16, 1))),
			(-11, -1, 1): (ValueLeaf((-12, -8, 1), (4, 8, 1)), ValueLeaf((-8, -4, 1), (8, 12, 1)), ValueLeaf((-4, 0, 1), (12, 16, 1))),
			(-7, -1, 1): (ValueLeaf((-8, -4, 1), (8, 12, 1)), ValueLeaf((-4, 0, 1), (12, 16, 1))),

			(-4, -1, 1): (ValueLeaf((-4, 0, 1), (12, 16, 1)),),
			(-13, -12, 1): (ValueLeaf((-16, -12, 1), (0, 4, 1)),),
			(-16, -15, 1): (ValueLeaf((-16, -12, 1), (0, 4, 1)),),
			(-9, -7, 1): (ValueLeaf((-12, -8, 1), (4, 8, 1)), ValueLeaf((-8, -4, 1), (8, 12, 1))),
		}
		self._testIndex(index, matrix, src)

	def testsClosest(self) -> None:
		src = ((0, 3, 1), (6, 7, 1), (12, 16, 1), (16, 20, 1))
		testMatrix = {
			(-10, -5, 1): ((0, 3, 1), (0, 0), 5),
			(100, 101, 1): ((16, 20, 1), (1, 1), 80),
			(5, 6, 1): ((6, 7, 1), (0, 1), 0),
			(4, 5, 1): ((0, 3, 1), (0, 0), 1),
			(4, 6, 1): ((6, 7, 1), (0, 1), 0),  # depends on python impl of `min`, may be (0, 3, 1)
		}

		for ctorSrc in isInstArg:
			t = self.__class__.indexerCtor([ctorSrc(*el) for el in src], None)

			for ctorQuery in isInstArg:
				for q, expectedRes in testMatrix.items():
					q = ctorQuery(*q)
					with self.subTest(q=q):
						expectedRes = [FuzzySingleLookupResult(self.__class__._genResItem(expectedRes[0], ctorSrc), *expectedRes[1:3], q)]
						res = t.get_closest(q)
						self.assertEqual(res, expectedRes)

	@classmethod
	def _genTestLenCombs(cls, initTree):
		l = len(initTree)
		for i in range(l - 1):
			for comb in itertools.combinations(range(l), i):
				selector = [True] * len(initTree)
				for el in comb:
					selector[el] = False
				yield (initTree.__class__(itertools.compress(initTree, selector)), l - i)

	def testsLen(self) -> None:
		initTree = ((0, 3, 1), (6, 7, 1), (12, 16, 1), (16, 20, 1))
		testMatrix = dict(self.__class__._genTestLenCombs(initTree))

		for challenge, response in testMatrix.items():
			with self.subTest(challenge=challenge, response=response):
				self.assertEqual(response, len(self.__class__.indexerCtor(cnss(range, challenge))))

	def testsSetAttr(self):
		setElProto = (18, 22, 1)
		treeProto = ((0, 4, 1), (4, 8, 1), (8, 12, 1), (12, 16, 1))
		matrix = {
			((-16, -12, 1), ): (
				((-16, -12, 1), (-12, -8, 1), (-8, -4, 1), (-4, 0, 1)),
				(setElProto, (4, 8, 1), (8, 12, 1), (12, 16, 1))
			),
			((-12, -8, 1), ): (
				((-16, -12, 1), (-12, -8, 1), (-8, -4, 1), (-4, 0, 1)),
				((0, 4, 1), setElProto, (8, 12, 1), (12, 16, 1))
			),
			((-8, -4, 1), ): (
				((-16, -12, 1), (-12, -8, 1), (-8, -4, 1), (-4, 0, 1)),
				((0, 4, 1), (4, 8, 1), setElProto, (12, 16, 1))
			),
			((-4, 0, 1), ): (
				((-16, -12, 1), (-12, -8, 1), (-8, -4, 1), (-4, 0, 1)),
				((0, 4, 1), (4, 8, 1), (8, 12, 1), setElProto)
			),
			((-20, -16, 1), ): (
				((-20, -16, 1), (-16, -12, 1), (-12, -8, 1), (-8, -4, 1), (-4, 0, 1)),
				(setElProto, (0, 4, 1), (4, 8, 1), (8, 12, 1), (12, 16, 1))
			),
			((16, 20, 1), ): (
				((-16, -12, 1), (-12, -8, 1), (-8, -4, 1), (-4, 0, 1), (16, 20, 1)),
				((0, 4, 1), (4, 8, 1), (8, 12, 1), (12, 16, 1), setElProto)
			),
		}

		for ctor in isInstArg:
			src = cnss(ctor, treeProto)
			#ic(src)
			index = ctor(-16, 0, 1)
			setEl = ctor(*setElProto)
			for indices2set, etalonFlatStructure in matrix.items():
				t = RangesTree.build(index=index, data=src)
				etalonFlatStructure = cnss(ctor, etalonFlatStructure)
				for index2Set in indices2set:
					index2Set = ctor(*index2Set)
					t[index2Set] = setEl
					self.assertEqual(list(t[index2Set])[0].indexee, setEl)

				self.assertEqual((tuple(el.index for el in t), tuple(el.indexee for el in t)), etalonFlatStructure)


#@unittest.skip
class SeqTests(IndexTestsProto):
	indexerCtor = SliceSequence

	def testSequenceTrivial(self) -> None:
		rng = (7, -1, -1)
		src = [rng]
		index = rng

		def genTests():
			for i in range(*rng):
				for j in range(i, *rng[1:]):
					q = (i, j, rng[2])
					yield (q, (ValueLeaf(q, q),))

		#matrix = dict(genTests())
		matrix = {
			(2, 0, -1): (ValueLeaf((2, 0, -1), (2, 0, -1)),)
		}
		self._testIndex(index, matrix, src)

	def testSequenceDumb(self) -> None:
		src = [(0, 4, 1), (4, 8, 1), (8, 12, 1), (12, 16, 1)]
		index = (0, 16, 1)

		matrix = {
			(0, 8, 1): (ValueLeaf((0, 4, 1), (0, 4, 1)), ValueLeaf((4, 8, 1), (4, 8, 1))),

			(8, 16, 1): (ValueLeaf((8, 12, 1), (8, 12, 1)), ValueLeaf((12, 16, 1), (12, 16, 1))),

			(4, 12, 1): (ValueLeaf((4, 8, 1), (4, 8, 1)), ValueLeaf((8, 12, 1), (8, 12, 1))),
			(4, 8, 1): (ValueLeaf((4, 8, 1), (4, 8, 1)),),


			(1, 15, 1): (ValueLeaf((1, 4, 1), (1, 4, 1)), ValueLeaf((4, 8, 1), (4, 8, 1)), ValueLeaf((8, 12, 1), (8, 12, 1)), ValueLeaf((12, 15, 1), (12, 15, 1))),
			(5, 15, 1): (ValueLeaf((5, 8, 1), (5, 8, 1)), ValueLeaf((8, 12, 1), (8, 12, 1)), ValueLeaf((12, 15, 1), (12, 15, 1))),
			(9, 15, 1): (ValueLeaf((9, 12, 1), (9, 12, 1)), ValueLeaf((12, 15, 1), (12, 15, 1))),


			(12, 15, 1): (ValueLeaf((12, 15, 1), (12, 15, 1)),),
			(3, 4, 1): (ValueLeaf((3, 4, 1), (3, 4, 1)),),
			(0, 1, 1): (ValueLeaf((0, 1, 1), (0, 1, 1)),),
			(7, 9, 1): (ValueLeaf((7, 8, 1), (7, 8, 1)), ValueLeaf((8, 9, 1), (8, 9, 1))),
		}
		self._testIndex(index, matrix, src)

	def testSequenceEndianness(self) -> None:
		src = [(7, -1, -1), (15, 7, -1)]
		index = (15, -1, -1)

		matrix = {
			(15, -1, -1): (ValueLeaf((15, 7, -1), (7, -1, -1)), ValueLeaf((7, -1, -1), (15, 7, -1))),
		}
		self._testIndex(index, matrix, src)


if __name__ == "__main__":
	unittest.main()
