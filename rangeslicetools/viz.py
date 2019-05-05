import typing
import itertools
from collections import defaultdict

from .utils import SliceRangeListT, SliceRangeT, sdir, slen, snormalize
from .diff import SDiffAutomata, sdiff


def sviz(ranges: SliceRangeListT):
	"""Draws ranges with ASCII art."""
	c = len(ranges)
	ruler = ""
	scale = ""
	ranges = sorted(ranges, key=lambda r: r.start)
	res = sdiff(*ranges)
	res = sorted((p for p in res.items()), key=lambda p: snormalize(p[1]).start)
	minf = -float("inf")
	ruler = defaultdict(lambda: minf)

	def pointsAndD(r):
		d = sdir(r)
		r = snormalize(r)
		return r.start, r.stop, d

	offset = len(str(res[0][1].start))
	maxStartPxPos = offset
	maxEndPxPos = maxStartPxPos

	for state, r in res:
		sp, ep, d = pointsAndD(r)
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

	def drawArea(r, f, layer):
		sp, ep, d = pointsAndD(r)
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
		ll = len(layer)
		return layer + (curImg)

	for ln in range(len(ranges)):
		drawn = False
		layer = " " * offset
		for k, r in res:
			ls = k[ln]
			if not ls & S.entered or ls & S.exited:

				def lam(d):
					return "...", "...", "."

				layer = drawArea(r, lam, layer)
			else:

				def lam(d):
					if d >= 0:
						e = "~>]"
						s = "[~~"
					else:
						s = "[<~"
						e = "~~]"
					return e, s, "~"

				layer = drawArea(ranges[ln], lam, layer)
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
