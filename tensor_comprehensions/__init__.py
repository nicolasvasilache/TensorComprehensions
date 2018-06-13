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
from typing import List, Optional, Tuple, Union

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
from tensor_comprehensions.tclib import Tuner
from tensor_comprehensions.tclib import TunerConfig
from tensor_comprehensions.tclib import compile

compile_delim="""
################################################################################
################################################################################
"""

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

class TC(object):
    def __init__(
            self,
            tc: str,
            entry_point: str,
            cache_filename: Optional[str] = "",
            tuner_config: Optional[TunerConfig] = TunerConfig()
    ):
        self.tc = tc
        self.entry_point = entry_point
        self.compilation_cache = CompilationCache(self.tc)
        self.cache_filename = cache_filename
        self.tuner_config = tuner_config
        self.warn = False

    def __call__(
            self,
            *inputs: torch.Tensor,
            outputs: Optional[Tuple[torch.Tensor]] = (),
            unchecked: Optional[bool] = False) -> List[torch.Tensor]:
        r"""
        Runs the defined TC function on given inputs. The TC must have been
        previously compiled for the specific input sizes or a compilation
        process will automatically start; which may kick off a whole tuning
        search.
        To avoid a longer tuning search, one can:
           1. [preferred] create your TC with a cache_filename (or call
              set_cache_filename). The file must be pre-populated by previous
              tuning runs. Once good solutions have been found they persist
              to disk;
           2. set the TC's tuner_config object to better control the tuning
              process. You can always abort tuning with Ctrl+C once good
              enough performance is reached and the best options will be
              appended to cache_filename;
           3. explicitly call compile(*inputs, mapping_options=...) to trigger
              compilation with the options you want to use and memoize them;

        Args:
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

        # Ensure compilation has happened, worst case be loud and kick a
        # tuning run.
        self.warn = True # implementing a context manager looks overkill here..
        self.compile(*inputs)
        self.warn = False

        inputs = tuple(inputs) if len(inputs) > 1 else tuple(inputs, )
        if unchecked:
            return self.compilation_cache.unchecked_run(
                self.entry_point, inputs, outputs)

        return self.compilation_cache.run(self.entry_point, inputs, outputs)

    def set_cache_filename(self, cache_filename : str):
        r"""Sets the backing cache_filename for this TC."""
        self.cache_filename = cache_filename

    def set_tuner_config(self, tuner_config : TunerConfig):
        r"""Sets the tuner_config for this TC."""
        self.tuner_config = tuner_config

    def compile(
            self,
            *inputs : torch.Tensor,
            mapping_options : Optional[Union[str, MappingOptions]] = None,
            autotune : Optional[bool] = False):
        already_compiled = self.compilation_cache.is_compiled(
            self.entry_point, inputs)

        mapping_options = (
            MappingOptions(mapping_options)
            if isinstance(mapping_options, str) else mapping_options)

        if not already_compiled and self.warn:
            print(
                compile_delim +
                "TC {} was not explicitly compiled ".format(self.entry_point) +
                "for inputs of sizes:\n  {}\n".format(
                    "".join("{} ".format(i.size().__str__()) for i in inputs)) +
                "TC.compile called with:\n  autotune={}".format(autotune) +
                ",\n  mapping_options={},\n  cache_filename={}".format(
                    mapping_options, self.cache_filename))

        if mapping_options is not None and self.warn:
            print(compile_delim +
                  "TC {} with user options for inputs of sizes:\n  {}".format(
                      self.entry_point,
                      "".join("{}, ".format(i.size().__str__()) for i in inputs)
                  ))

        if not already_compiled and mapping_options is None:
            force_tune = False
            cache = MappingOptionsCache(self.cache_filename)
            base_options_list = cache.load(self.tc, self.entry_point, inputs, 1)
            if len(base_options_list) > 0:
                if self.warn:
                    print("""Using MappingOptions retrieved from {}""".format(
                        self.cache_filename))
                mapping_options = base_options_list[0]
            else:
                if self.warn:
                    print("""Using naive options => force autotune=True""")
                mapping_options = MappingOptions('naive')
                force_tune = True

            if autotune or force_tune:
                if self.warn:
                    print(
                        "Starting a tuning run, abort it with Ctrl+C when "+
                        "performance is satisfactory.\n You can bypass " +
                        "tuning by either:\n\t" +
                        "1. calling compile explicitly and providing "+
                        "MappingOptions manually, or\n\t" +
                        "2. by passing a compilation cache filename in which "+
                        "best MappingOptions\n        " +
                        "resulting from tuning can be " +
                        "persisted across processes.")
                mapping_options = self.tune(
                    *inputs,
                    starting_options=mapping_options,
                    store_to_cache=True)

        # Compile best options to set the executor for the current
        # (entry point, inputs)
        if mapping_options is not None:
            start = time.clock()
            if self.warn:
                print("Start compiling TC {} with inputs of sizes:  \n{}".format(
                    self.entry_point,
                    "".join("{}, ".format(i.size().__str__()) for i in inputs)))
            self.compilation_cache.compile(
                self.entry_point, inputs, mapping_options)
            if self.warn:
                print(("Done compiling TC (compile time: {}s)" +
                       compile_delim).format(int(time.clock() - start)))

    def tune(self,
             *inputs : torch.Tensor,
             starting_options : Optional[Union[str, MappingOptions]] = None,
             load_from_cache : Optional[bool] = False,
             store_to_cache : Optional[bool] = False) -> MappingOptions:
        r"""Tunes the defined TC function for given inputs.

        This function potentially interacts with the TC's backing cache file by
        loading from and storing to cache. The MappingOptions from which tuning
        starts is determined from the optional parameters as follows:
            1. if starting_options is specified, it takes precedence;
            2. otherwise if load_from_cache is True, the best options for the
               current TC and input sizes are fetched from the backing
               self.cache_filename;
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
                the best MappingOptions by loading from self.cache_filename.
                If starting_options is not specified, and loading fails to
                recover a MappingOptions from the cache file, then tuning starts
                from MappingOptions('naive')
            store_to_cache: Optionally store the best result by appending it to
                the backing cache file

        Returns:
            MappingOptions: the best mapping options found during this run.
        """

        inputs = tuple(inputs) if len(inputs) > 1 else tuple(inputs, )
        cache = MappingOptionsCache(self.cache_filename)

        base_options = (
            MappingOptions(starting_options)
            if isinstance(starting_options, str) else starting_options)
        if base_options is None and load_from_cache:
            loaded = cache.load(self.tc, self.entry_point, inputs, 1)
            if len(loaded) > 0:
                base_options = loaded[0]
        if base_options is None:
            base_options = MappingOptions('naive')

        # TODO: This is still an implicit store behavior in the C++ API,
        #     make it explicit...
        tuner = Tuner(self.tc, self.cache_filename if store_to_cache else "")
        best_options = tuner.tune(
            self.entry_point,
            inputs,
            base_options,
            self.tuner_config
        )

        return best_options

    def tune_and_compile(
            self,
            *inputs : torch.Tensor,
            starting_options : Optional[Union[str, MappingOptions]] = None,
            load_from_cache : Optional[bool] = False,
            store_to_cache : Optional[bool] = False) -> MappingOptions:
        r"""Calls tune then compiles with the best options."""
        best = self.tune(
            *inputs,
            starting_options=starting_options,
            load_from_cache=load_from_cache,
            store_to_cache=store_to_cache)
        self.compile(*inputs, mapping_options=best)
        return best

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
    def __init__(self, forward_fun, backward_fun=None):
        self.forward_fun = forward_fun
        self.backward_fun = backward_fun

    def __call__(self, *inputs):
        return TcFunction.apply(self.forward_fun, self.backward_fun, *inputs)

def define(tc, entry_point, **kwargs):
    return TC(tc, entry_point, **kwargs)

def define_with_autograd(forward_fun, backward_fun=None):
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
