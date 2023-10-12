import sys
import types
import typing
from functools import wraps

from warnings import warn

warn("We have moved from M$ GitHub to https://codeberg.org/KOLANICH-libs/rangeslicetools.py, read why on https://codeberg.org/KOLANICH/Fuck-GuanTEEnomo .")

from . import diff, utils


def _createWrapped(f: typing.Callable) -> typing.Callable:
	@wraps(f)
	def f1(*args, **kwargs):
		return tuple(f(*args, **kwargs))

	f1.__annotations__["return"] = utils.SliceRangeListT
	return f1


def _wrapModuleProp(module, k, v, _all_) -> None:
	if k[0] != "_":
		_all_.append(k)

	if k[0] == "s" and k[-1] == "_":
		if "return" not in v.__annotations__:
			raise ValueError("Annotate the return type in " + v.__qualname__ + "!")

		modName = k[:-1]
		if v.__annotations__["return"] is module.SliceRangeSeqT and modName not in module.__dict__:
			module.__dict__[modName] = _createWrapped(v)
			_all_.append(modName)

	module.__dict__[k] = v


def _wrap(module) -> None:
	_all_ = getattr(module, "__all__", None)
	if _all_ is None:
		_all_ = []
		for k, v in tuple(module.__dict__.items()):
			_wrapModuleProp(module, k, v, _all_)
			_all_.append(k)
	else:
		_all_ = list(_all_)
		for k in list(_all_):
			v = getattr(module, k)
			_wrapModuleProp(module, k, v, _all_)
		_all_ = tuple(sorted(_all_))

	module.__all__ = tuple(_all_)

	sys.modules[module.__name__] = module


_wrap(utils)
_wrap(diff)


# pylint: disable=wrong-import-position
from .utils import *  # noqa
from .diff import *  # noqa
from .tree import *  # noqa
from .viz import *  # noqa
