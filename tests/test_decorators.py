import pytest

from costly.decorators import costable

@costable("blabla")
def f_noargs():
    return description
    
@costable("blabla")
def f_arg(description: list[str]):
    return description

@costable("blabla2")
def f_argopt(description: list[str] | None = None):
    return description

@costable("blabla3")
def f_args(*args):
    return args

@costable("blabla4")
def f_kwargs(**kwargs):
    return kwargs.get("description", None)

@costable("blabla5")
def f_arg_args(description: list[str], *args):
    return description, args

@costable("blabla6")
def f_arg_kwargs(description: list[str], **kwargs):
    return description

@costable("blabla7")
def f_arg_args_kwargs(description: list[str], *args, **kwargs):
    return description

@costable("blabla8")
def f_argopt_args_kwargs(description: list[str] | None = None, *args, **kwargs):
    return description

def test_costable():
    v = f_noargs()
    if not v == ["blabla"]:
        pytest.fail(f"f_noargs() returned {v} instead of ['blabla']")
    v = f_arg(["a"])
    if not v == ["blabla", "a"]:
        pytest.fail(f"f_arg(['a']) returned {v} instead of ['blabla', 'a']")
    v = f_argopt(["a"])
    if not v == ["blabla", "a"]:
        pytest.fail(f"f_argopt(['a']) returned {v} instead of ['blabla', 'a']")
    v = f_argopt()
    if not v == ["blabla"]:
        pytest.fail(f"f_argopt() returned {f_argopt()} instead of ['blabla']")
    if not f_args("a") == ("a",):
        pytest.fail(f"f_args('a') returned {f_args('a')} instead of ('a',)")
    v = f_kwargs(description="a")
    if not v == "a":
        pytest.fail(f"f_kwargs(description='a') returned {v} instead of 'a'")
    v = f_arg_args(["a"], "b")
    if not v == (["a"], "b"):
        pytest.fail(f"f_arg_args(['a'], 'b') returned {v} instead of (['a'], 'b')")
    v = f_arg_kwargs(["a"], description="b")
    if not v == ["a"]:
        pytest.fail(f"f_arg_kwargs(['a'], description='b') returned {v} instead of ['a']")
    v = f_arg_args_kwargs(["a"], "b", description="c")
    if not v == ["a"]:
        pytest.fail(f"f_arg_args_kwargs(['a'], 'b', description='c') returned {v} instead of ['a']")
    v = f_argopt_args_kwargs(["a"], "b", description="c")
    if not v == ["a"]:
        pytest.fail(f"f_argopt_args_kwargs(['a'], 'b', description='c') returned {v} instead of ['a']")
    v = f_argopt_args_kwargs(description="a")
    if not v == ["blabla", "a"]:
        pytest.fail(f"f_argopt_args_kwargs(description='a') returned {v} instead of ['blabla', 'a']")
    v = f_argopt_args_kwargs()
    if not v == ["blabla"]:
        pytest.fail(f"f_argopt_args_kwargs() returned {f_argopt_args_kwargs()} instead of ['blabla']")
    v = f_argopt_args_kwargs(["a"], "b", description="c")
    if not v == ["a"]:
        pytest.fail(f"f_argopt_args_kwargs(['a'], 'b', description='c') returned {v} instead of ['a']")


