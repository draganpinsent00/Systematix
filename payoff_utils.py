import ast
import types
from typing import Callable
import numpy as np
import multiprocessing as mp
import traceback
import os
import sys
try:
    import resource
    _HAS_RLIMIT = True
except Exception:
    _HAS_RLIMIT = False
from utils.payoff_helpers import discount_factor, terminal_call, terminal_put, asian_arithmetic, path_stats


class _SafeChecker(ast.NodeVisitor):
    """AST checker that rejects unsafe constructs for user payoff code."""
    SAFE_BUILTINS = {'max', 'min', 'abs', 'round'}
    SAFE_NAMES = {'np', 'math'} | SAFE_BUILTINS | {'paths'}

    def __init__(self):
        super().__init__()
        self.found_func = False

    def visit_Import(self, node):
        # Allow only: import numpy as np
        for alias in node.names:
            if not (alias.name == 'numpy' and alias.asname == 'np'):
                raise ValueError('Only "import numpy as np" is allowed in custom payoff code')

    def visit_ImportFrom(self, node):
        raise ValueError('from-import statements are not allowed in custom payoff code')

    def visit_Global(self, node):
        raise ValueError('Global statements not allowed')

    def visit_Nonlocal(self, node):
        raise ValueError('Nonlocal statements not allowed')

    def visit_Call(self, node):
        # allow calls to np.* or safe builtins
        func = node.func
        if isinstance(func, ast.Attribute):
            # ensure base is Name 'np' (we only expose numpy)
            if not (isinstance(func.value, ast.Name) and func.value.id == 'np'):
                raise ValueError('Only numpy (np) functions are allowed in calls')
        elif isinstance(func, ast.Name):
            if func.id not in self.SAFE_BUILTINS:
                raise ValueError(f'Calling function {func.id} is not allowed')
        else:
            raise ValueError('Unsupported call expression in custom payoff')
        # visit args
        for a in node.args:
            self.visit(a)
        for kw in node.keywords:
            self.visit(kw.value)

    def visit_Attribute(self, node):
        # allow attribute access on np only
        if isinstance(node.value, ast.Name):
            if node.value.id != 'np':
                raise ValueError('Attribute access only allowed on np (numpy)')
        else:
            self.generic_visit(node)

    def visit_Name(self, node):
        # disallow dunder and unknown names except 'paths' and numpy names
        if node.id.startswith('__'):
            raise ValueError('Use of dunder names not allowed')
        # Names used as variables are allowed; we just block suspicious ones in other nodes

    def visit_FunctionDef(self, node):
        # require function named custom_payoff
        if node.name != 'custom_payoff':
            raise ValueError("Custom function must be named 'custom_payoff'")
        # only one function
        if self.found_func:
            raise ValueError('Only a single function definition is allowed')
        self.found_func = True
        # no decorators
        if node.decorator_list:
            raise ValueError('Function decorators are not allowed')
        # visit body
        for b in node.body:
            self.visit(b)

    def visit_ClassDef(self, node):
        raise ValueError('Class definitions are not allowed')

    def visit_Exec(self, node):  # pragma: no cover - Py3 doesn't have Exec node
        raise ValueError('exec not allowed')

    # Allow common nodes: BinOp, UnaryOp, Compare, Return, Expr, Subscript, Index, Constant, IfExp
    def generic_visit(self, node):
        # whitelist a set of nodes;otherwise delegate
        allowed = (
            ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.Expr,
            ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.If, ast.IfExp, ast.Subscript,
            ast.Index, ast.Slice, ast.Load, ast.Store, ast.Tuple, ast.List, ast.ListComp,
            ast.Constant, ast.Name, ast.Attribute, ast.Call,
            # operator and comparator nodes
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
            ast.UAdd, ast.USub,
            ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE
        )
        if isinstance(node, allowed):
            super().generic_visit(node)
        else:
            raise ValueError(f'Node type {type(node).__name__} is not allowed in custom payoff')


def safe_compile_payoff(code: str) -> Callable:
    """Safely compile user-supplied payoff code.

    The user must supply a single function named `custom_payoff(paths)` that returns a 1-D numpy array.
    The code may call numpy functions via `np.<func>` and safe builtins (max, min, abs, round).
    """
    if not isinstance(code, str):
        raise ValueError('Code must be a string')
    # parse
    try:
        tree = ast.parse(code, mode='exec')
    except SyntaxError as e:
        raise ValueError(f'Syntax error in custom payoff code: {e}')

    checker = _SafeChecker()
    checker.visit(tree)
    if not checker.found_func:
        raise ValueError("No function named 'custom_payoff' found in code")

    # compile in a restricted namespace
    glbs = {'np': np}
    locs = {}
    try:
        compiled = compile(tree, filename='<custom_payoff>', mode='exec')
        exec(compiled, glbs, locs)
    except Exception as e:
        raise ValueError(f'Error compiling custom payoff: {e}')

    if 'custom_payoff' not in locs:
        raise ValueError("custom_payoff not found after compilation")
    fn = locs['custom_payoff']
    if not callable(fn):
        raise ValueError('custom_payoff is not callable')

    # wrapper that enforces numpy array output and 1-D shape
    def wrapper(paths: np.ndarray, **context) -> np.ndarray:
        # temporarily inject provided context into function globals so user code can reference S0,K etc.
        old_globals = {}
        g = fn.__globals__
        try:
            for k, v in context.items():
                if k in g:
                    old_globals[k] = g[k]
                else:
                    old_globals[k] = None
                g[k] = v
            res = fn(paths)
        finally:
            # restore originals
            for k, old in old_globals.items():
                if old is None:
                    # remove the injected name
                    try:
                        del g[k]
                    except KeyError:
                        pass
                else:
                    g[k] = old

        res = np.asarray(res)
        if res.ndim != 1:
            raise ValueError('custom_payoff must return a 1-D array of payoffs')
        if res.shape[0] != paths.shape[0]:
            raise ValueError('custom_payoff must return an array of length n_paths')
        # check for nan or inf
        if np.isnan(res).any() or np.isinf(res).any():
            raise ValueError('custom_payoff returned NaN or infinite values; please check code and inputs')
        return res

    return wrapper


def _worker_exec(code_str: str, paths, context, q: mp.Queue):
    """Worker to compile and execute the custom payoff inside a child process."""
    try:
        # apply resource limits if possible
        if _HAS_RLIMIT:
            # Memory limit (virtual address space)
            soft_mem, hard_mem = 100 * 1024 * 1024, 200 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (soft_mem, hard_mem))
            # CPU time limit in seconds (wall CPU seconds for the process)
            soft_cpu, hard_cpu = 5, 10
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (soft_cpu, hard_cpu))
            except Exception:
                # some platforms may not allow changing CPU rlimit
                pass
            # Disallow creating large files: file size limit
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
            except Exception:
                pass
            # Reduce number of open files
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (16, 64))
            except Exception:
                pass
        # compile safely
        fn = safe_compile_payoff(code_str)
        # inject curated helpers into function globals
        g = fn.__globals__
        helpers = {
            'discount_factor': discount_factor,
            'terminal_call': terminal_call,
            'terminal_put': terminal_put,
            'asian_arithmetic': asian_arithmetic,
            'path_stats': path_stats,
            'np': __import__('numpy')
        }
        old = {k: g.get(k, None) for k in helpers}
        g.update(helpers)
        try:
            res = fn(paths, **context)
        finally:
            # restore globals
            for k, v in old.items():
                if v is None:
                    g.pop(k, None)
                else:
                    g[k] = v
        q.put(('OK', res))
    except Exception as e:
        tb = traceback.format_exc()
        q.put(('ERR', (str(e), tb)))


def run_payoff_in_sandbox(code: str, paths: np.ndarray, context: dict = None, timeout: float = 2.0):
    """Run the custom payoff code in a separate process with a timeout.

    Returns the payoff 1-D numpy array or raises ValueError on user error/timeout.
    """
    if context is None:
        context = {}
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker_exec, args=(code, paths, context, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(0.1)
        raise TimeoutError('Custom payoff execution exceeded time limit')
    if q.empty():
        raise RuntimeError('Custom payoff worker exited without result')
    status, payload = q.get()
    if status == 'OK':
        return np.asarray(payload)
    else:
        msg, tb = payload
        raise ValueError(f'Error in custom payoff: {msg}\nTraceback:\n{tb}')
