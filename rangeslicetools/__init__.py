import typing
import sys
import types
from functools import wraps

from . import utils
from . import diff


def _createWrapped(f: typing.Callable) -> typing.Callable:
	@wraps(f)
	def f1(*args, **kwargs):
		return tuple(f(*args, **kwargs))

	f1.__annotations__["return"] = utils.SliceRangeListT
	return f1


def _wrapModuleProp(module, k, v, all) -> None:
	if k[0] != "_":
		all.append(k)

	if k[0] == "s" and k[-1] == "_":
		if "return" not in v.__annotations__:
			raise ValueError("Annotate the return type in " + v.__qualname__ + "!")

		modName = k[:-1]
		if v.__annotations__["return"] is module.SliceRangeSeqT and modName not in module.__dict__:
			module.__dict__[modName] = _createWrapped(v)
			all.append(modName)

	module.__dict__[k] = v


def _wrap(module) -> None:
	all = getattr(module, "__all__", None)
	if all is None:
		all = []
		for k, v in tuple(module.__dict__.items()):
			_wrapModuleProp(module, k, v, all)
			all.append(k)
	else:
		all = list(all)
		for k in list(all):
			v = getattr(module, k)
			_wrapModuleProp(module, k, v, all)
		all = tuple(sorted(all))

	module.__all__ = tuple(all)

	sys.modules[module.__name__] = module


_wrap(utils)
_wrap(diff)

# pylint: disable=wrong-import-position
from .utils import *  # noqa
from .diff import *  # noqa
from .tree import *  # noqa
from .viz import *  # noqa
