import typing
from collections import defaultdict

from .diff import SDiffAutomata, sdiff
from .utils import SliceRangeListT, SliceRangeT, sdir, slen, snormalize


def _pointsAndD(r: range) -> typing.Tuple[int, int, int]:
	d = sdir(r)
	r = snormalize(r)
	return r.start, r.stop, d


def _drawArea(r: range, f: typing.Callable, layer: str, ruler: typing.Mapping[int, int]) -> str:
	sp, ep, d = _pointsAndD(r)
	startX = ruler[sp]
	endX = ruler[ep]
	pixLen = endX - startX

	nS = str(ep - sp)

	e, s, filler = f(d)

	iL = len(e) + len(s) + len(nS)
	margin = pixLen - iL
	mh = margin // 2
	msh = margin - mh

	if d >= 0:
		e = filler * mh + e
		s = s + filler * msh
	else:
		s += filler * mh
		e = filler * msh + e

	curImg = s + nS + e
	return layer + (curImg)


def sviz(ranges: SliceRangeListT) -> str:
	"""Draws ranges with ASCII art."""
	c = len(ranges)
	ruler = ""
	scale = ""
	ranges = sorted(ranges, key=lambda r: r.start)
	res = sdiff(*ranges)
	res = sorted((p for p in res.items()), key=lambda p: snormalize(p[1]).start)
	minf = -float("inf")
	ruler = defaultdict(lambda: minf)

	offset = len(str(res[0][1].start))
	maxStartPxPos = offset
	maxEndPxPos = maxStartPxPos

	for state, r in res:
		sp, ep, d = _pointsAndD(r)
		l = ep - sp

		numW = len(str(sp))
		maxStartPxPos = maxEndPxPos + numW + 3  # for arrow
		ruler[sp] = max(maxStartPxPos - numW // 2, ruler[sp])

		maxEndPxPos = maxStartPxPos + len(str(l))
		maxEndPxPos += 3  # for arrow
		ruler[ep] = max(maxEndPxPos, ruler[sp])

	e = ""
	s = ""
	layers = []

	S = SDiffAutomata.State

	for ln in range(len(ranges)):
		drawn = False
		layer = " " * offset
		for k, r in res:
			ls = k[ln]
			if not ls & S.entered or ls & S.exited:

				def lam(d):
					return "...", "...", "."

				layer = _drawArea(r, lam, layer, ruler)
			else:

				def lam(d):
					if d >= 0:
						e = "~>]"
						s = "[~~"
					else:
						s = "[<~"
						e = "~~]"
					return e, s, "~"

				layer = _drawArea(ranges[ln], lam, layer, ruler)
				layers.append(layer)
				break

	layers = ("".join(l) for l in layers)

	ruler = sorted((p for p in ruler.items()), key=lambda p: p[0])
	rulerScale = " " * offset
	rulerImg = ""
	for x, px in ruler:
		margin = px - len(rulerImg)
		ns = str(x)
		rulerImg += ns + "." * margin
		rulerScale += "|" + "." * (margin + len(ns) // 2)
	res = "\n".join(layers)
	res += "\n" + rulerScale + "\n" + rulerImg
	return res
