# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
from typing import Callable, Iterable, List, Optional, Tuple, Union

import time

# Importing pytorch before trying to dlopen tclib is currently required
# because of:
#   https://github.com/pytorch/pytorch/issues/6097
# This probably requires a patch on the pytorch side to remove the dependency
import torch

from tensor_comprehensions.tclib import logtostderr
from tensor_comprehensions.tclib import debug_lang
from tensor_comprehensions.tclib import debug_halide
from tensor_comprehensions.tclib import debug_tc_mapper
from tensor_comprehensions.tclib import debug_tuner
from tensor_comprehensions.tclib import dump_cuda

from tensor_comprehensions.tclib import CompilationCache
from tensor_comprehensions.tclib import MappingOptions
from tensor_comprehensions.tclib import MappingOptionsCache
from tensor_comprehensions.tclib import TcExecutor
from tensor_comprehensions.tclib import Tuner
from tensor_comprehensions.tclib import TunerConfig

import tensor_comprehensions.tclib as tclib

global __SILENT__
__SILENT__ = False

def assert_almost_equal(diff, inputs, operations=1, precision=1e-7):
    max_value = 0.0
    for inp in inputs:
        max_value = max(float(inp.abs().max()), max_value)
    max_diff = float(diff.abs().max())
    assert max_diff <= operations * precision * max_value, (
        ("error at relative precision: {}, #operations: {}, " +
         "max_value: {}, max_diff: {}").format(
            precision, operations, max_value, max_diff)
    )

class Executor(object):
    def __init__(self, executor: TcExecutor):
        self.executor = executor

    def __call__(
            self,
            *inputs: torch.Tensor,
            outputs: Optional[Tuple[torch.Tensor]] = (),
            unchecked: Optional[bool] = False) -> List[torch.Tensor]:

        inputs = tuple(inputs) if len(inputs) > 1 else tuple(inputs, )
        if unchecked:
            return self.executor.unchecked_run(inputs, outputs)

        return self.executor.run(inputs, outputs)

def compile(
        tc: str,
        entry_point: str,
        mapping_options: Union[str, MappingOptions],
        *inputs: torch.Tensor) -> Executor:
    mapping_options = (
        MappingOptions(mapping_options)
        if isinstance(mapping_options, str) else mapping_options)
    return Executor(tclib.compile(tc, entry_point, inputs, mapping_options))

def autotune(tc: str,
             entry_point: str,
             *inputs: torch.Tensor,
             starting_options: Optional[Union[str, MappingOptions]] = None,
             tuner_config: Optional[TunerConfig] = TunerConfig(),
             cache_filename: Optional[str] = None,
             load_from_cache: Optional[bool] = False,
             store_to_cache: Optional[bool] = False) -> MappingOptions:
    r"""Tunes the defined TC function for given inputs.

        This function potentially interacts with the TC's backing cache file by
        loading from and storing to cache. The MappingOptions from which tuning
        starts is determined from the optional parameters as follows:
            1. if starting_options is specified, it takes precedence;
            2. otherwise if load_from_cache is True, the best options for the
               current TC and input sizes are fetched from the backing
               cache_filename;
            3. if none of steps 1. or 2. above yield a MappingOptions,
               tuning starts from MappingOptions('naive')

        It is possible to obtain a reinforcement tuning behavior by tuning over
        multiple executions and specifying both load_from_cache and
        store_to_cache.
        It is recommended to only use a single cache file for all TCs and
        reinforce it over time.

        Args:
            inputs: PyTorch Tensors that TC should tune for. The inputs must be
                passed in the order they are also passed in the definition of
                the TC function.
            starting_options: MappingOptions from which tuning should start
                from. If specified, starting_options takes precedence.
            load_from_cache: In the absence of explicit starting_options, get
                the best MappingOptions by loading from cache_filename.
                If starting_options is not specified, and loading fails to
                recover a MappingOptions from the cache file, then tuning starts
                from MappingOptions('naive')
            store_to_cache: Optionally store the best result by appending it to
                the backing cache file

        Returns:
            MappingOptions: the best mapping options found during this run.
        """

    if load_from_cache or store_to_cache:
        assert cache_filename is not None, ("load_from_cache or store_to_cache" +
        " specified, must also specify cache_filename")

    inputs = tuple(inputs) if len(inputs) > 1 else tuple(inputs, )

    base_options = (
        MappingOptions(starting_options)
        if isinstance(starting_options, str) else starting_options)
    if base_options is None and load_from_cache:
        cache = MappingOptionsCache(cache_filename)
        loaded = cache.load(tc, entry_point, inputs, 1)
        if len(loaded) > 0:
            base_options = loaded[0]
    if base_options is None:
        base_options = MappingOptions('naive')

    # TODO: This is still an implicit store behavior in the C++ API,
    #     make it explicit...
    tuner = Tuner(tc, cache_filename if store_to_cache else "")
    return tuner.tune(
        entry_point,
        inputs,
        base_options,
        tuner_config)

def autotune_and_compile(
        tc: str,
        entry_point: str,
        *inputs: torch.Tensor,
        starting_options: Optional[Union[str, MappingOptions]] = None,
        tuner_config: Optional[TunerConfig] = TunerConfig(),
        cache_filename: Optional[str] = None,
        load_from_cache: Optional[bool] = False,
        store_to_cache: Optional[bool] = False) -> Executor:
    r"""Calls autotune, compiles with best options then returns an Executor."""
    best = autotune(
        tc,
        entry_point,
        *inputs,
        starting_options=starting_options,
        tuner_config=tuner_config,
        cache_filename=cache_filename,
        load_from_cache=load_from_cache,
        store_to_cache=store_to_cache)
    if best is None:
        return None
    return compile(tc, entry_point, best, *inputs)

class TC(object):
    @staticmethod
    def generate_naive_options():
        def generate(tc_obj: TC,
                     entry_point: str,
                     *inputs: torch.Tensor) -> MappingOptions:
            return MappingOptions('naive')

        return generate

    @staticmethod
    def generate_options_load_from_cache(cache_filename: str):
        def generate(tc_obj: TC,
                     entry_point: str,
                     *inputs: torch.Tensor) -> MappingOptions:
            cache = MappingOptionsCache(cache_filename)
            loaded = cache.load(tc_obj.tc, entry_point, inputs, 1)
            if len(loaded) > 0:
                return loaded[0]
            return None

        return generate

    @staticmethod
    def generate_options_from_tuning(
            starting_options: Optional[Union[str, MappingOptions]] = None,
            tuner_config: TunerConfig = TunerConfig(),
            cache_filename: Optional[str] = None,
            load_from_cache: Optional[bool] = False,
            store_to_cache: Optional[bool] = False):
        def generate(tc_obj: TC,
                     entry_point: str,
                     *inputs: torch.Tensor) -> MappingOptions:
            return autotune(
                tc_obj.tc,
                entry_point,
                *inputs,
                starting_options=starting_options,
                tuner_config=tuner_config,
                cache_filename=cache_filename,
                load_from_cache=load_from_cache,
                store_to_cache=store_to_cache)

        return generate

    def __init__(
            self,
            tc: str,
            generate_implicit_mapping_options
            #: Callable[[TC, str, Iterable[torch.Tensor]], MappingOptions]
            # TC not yet defined at this point, punting on type annotation
    ):
        self.tc = tc
        self.generate_implicit_mapping_options = generate_implicit_mapping_options
        self.compilation_cache = CompilationCache(self.tc)
        # Make each TC def in the tc str a method of the TC object so we can:
        #     T = tc.define("def add() ...")
        #     T.add()
        #
        def make_closure(obj: TC, tc_def_name: str):
            def fun(*inputs: torch.Tensor,
                    outputs: Optional[Tuple[torch.Tensor]] = (),
                    unchecked: Optional[bool] = False) -> List[torch.Tensor] :
                return obj.__call__(
                    tc_def_name, *inputs, outputs=outputs, unchecked=unchecked)

            return fun

        for tc_def in tclib.parse_defs(self.tc):
            self.__setattr__(tc_def, make_closure(self, tc_def))

    def __call__(
            self,
            entry_point: str,
            *inputs: torch.Tensor,
            outputs: Optional[Tuple[torch.Tensor]] = (),
            unchecked: Optional[bool] = False) -> List[torch.Tensor]:
        r"""
        Runs the defined TC function on given inputs. The TC must have been
        previously compiled for the specific input sizes or a compilation
        process will automatically start.

        Args:
            entry_point: the name of the TC def to be executed

            inputs: tensors that TC takes as input arguments. The inputs must
                be passed in the order of the definition of the TC function.

            outputs: tensors the TC writes to in-place. If output tensors are
                omitted, new outputs will be allocated and returned at each
                call.

            unchecked: run in low-latency unchecked mode where failsafe size
                and stride checks are omitted.

        Returns
            List[torch.Tensor]: the tensors returned by the TC. If outputs are
                not specified then new allocations take place. Otherwise the TC
                writes in-place and returns the same tensors it received as
                outputs.
        """

        # Locally scoped implicit compilation
        def __implicit_compile__(tc_obj: TC,
                                 entry_point: str,
                                 *inputs: torch.Tensor):
            already_compiled = tc_obj.compilation_cache.is_compiled(
                entry_point, inputs)

            if already_compiled:
                return

            if not __SILENT__:
                print(
                    "TC \"{}\" was not explicitly compiled for ".format(entry_point) +
                    "inputs of sizes:\n  {}\n".format("".join("{}, ".format(
                        i.size().__str__()) for i in inputs)) +
                    "....Generate implicit MappingOptions")

            mapping_options = tc_obj.generate_implicit_mapping_options(
                tc_obj, entry_point, *inputs)

            assert mapping_options is not None, (
                "No options found for TC {} ".format(entry_point) +
                "with inputs of sizes:\n  {}\n".format(
                    "".join("{} ".format(i.size().__str__()) for i in inputs)))

            # Compile best options to set the executor for the current
            #     (entry point, inputs)
            start = time.clock()
            tc_obj.compilation_cache.compile(
                entry_point, inputs, mapping_options)
            if not __SILENT__:
                print(
                    "Done compiling TC \"{}\" (compile time: {}ms)".format(
                        entry_point, int((time.clock() - start) * 10 ** 3)))

        __implicit_compile__(self, entry_point, *inputs)

        inputs = tuple(inputs) if len(inputs) > 1 else tuple(inputs, )
        if unchecked:
            return self.compilation_cache.unchecked_run(
                entry_point, inputs, outputs)

        return self.compilation_cache.run(entry_point, inputs, outputs)

def define(tc, generate_implicit_mapping_options):
    return TC(tc, generate_implicit_mapping_options)

class TcFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, forward_fun, backward_fun, *inputs):
        ctx.backward_fun = backward_fun
        ctx.save_for_backward(*inputs)
        return tuple(forward_fun(*inputs))

    @staticmethod
    def backward(ctx, *gradients):
        if ctx.backward_fun is not None:
            inputs = tuple(ctx.saved_tensors) + tuple(
                t.contiguous() for t in gradients)
            # PyTorch convention: need an extra None return for each of
            # forward_fun and backward_fun,
            return (None, None, *tuple(ctx.backward_fun(*inputs)))

        return (None, )

class TCWithAutograd(object):
    def __init__(self, forward_fun, backward_fun):
        self.forward_fun = forward_fun
        self.backward_fun = backward_fun

    def __call__(self, *inputs):
        return TcFunction.apply(self.forward_fun, self.backward_fun, *inputs)

def define_with_autograd(forward_fun, backward_fun):
    return TCWithAutograd(forward_fun, backward_fun)

__all__ = [
    # Debugging functions, pass True to activate
    'logtostderr',
    'debug_lang',
    'debug_halide',
    'debug_tc_mapper',
    'debug_tuner',
    'dump_cuda',
    # Functions exposed by the tclib
    'compile',
    'autotune',
    'autotune_and_compile',
    # Classes exposed by the tclib
    'CompilationCache',
    'MappingOptions',
    'MappingOptionsCache',
    'Tuner',
    'TunerConfig',
    # Python-side functionality
    'assert_almost_equal',
    'define',
    'define_with_autograd',
    'TC',
    'TCWithAutograd',
]
