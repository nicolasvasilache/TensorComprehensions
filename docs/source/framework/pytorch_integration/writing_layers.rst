Writing TC operations
=====================

To create a CUDA kernel implementing an operation backed by TC, one can:

1. Create a TC object by calling :code:`tc.define`
2. Create input torch tensors
3. Compile or tune the TC

When running such a TC on the inputs created in step 2, the backend ensures
the TC is compiled and memoized for the given input tensor sizes.
Calling the :code:`TC` object returned by :code:`tc.define` executes the
corresponding operation and returns a list of outputs.
If the operation has already been compiled, in the following runs, the TC
backend will reuse the memoized compilation result and run the operation
directly.

Example
-------

The following example demonstrates the steps above.
Note that an explicit compile step is recommended.
The only user-facing :code:`MappingOptions` object that can be
constructed is a :code:`naive` object by calling :code:`tc.MappingOptions('naive')`.
At this time there is no notion of a default :code:`MappingOptions` object.
Instead one should use the autotuner to perform an evolutionary search
starting from an initial :code:`MappingOptions` object and return a better
:code:`MappingOptions` object for a given TC function and sizes (more on this
below).

    .. note::

       The fallback parameter is optional, however a TC constructed without a
       fallback must be explicitly tuned or compiled beforehand. Trying to
       call a TC that hasn't been compiled or tuned and that was constructed
       without a fallback will result in an error.

    .. code-block:: python

        import tensor_comprehensions as tc
        import torch
        mm = """
        def matmul(float(M, K) A, float(K, N) B) -> (C) {
            C(m, n) +=! A(m, r_k) * B(r_k, n)
        }
        """
        # the `entry_point` should match the definition name in `mm`
        matmul = tc.define(mm, entry_point="matmul")
        A, B = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
        # the following call will trigger compilation and memoization and return a
        # list of output tensors
        matmul.compile(A, B, mapping_options='naive')
        C, = matmul(A, B)
        # a subsequent call to the same TC with the same sizes will not re-trigger
        # compilation and memoization
        C, = matmul(A, B)
        # optionally, a tuple of properly-sized output tensors can be passed and
        # the kernel will use them as outputs
        C, = matmul(A, B, outputs=(out, ))

Specifying MappingOptions
-----------------------------

There are three ways to construct :code:`MappingOptions` when defining a TC:

* **Naive MappingOptions**:

  * :code:`naive`: this is provided to create a basic mapping strategy with
    3-D tiling by 32x32x32, mapping to 256x256 blocks 32x8 threads. This
    should by no means be considered a good baseline but just a point to
    get started using TC. Once a correct TC is written, we recommend either
    using options loaded from a :code:`MappingOptionsCache` or resulting from
    a tuning run.

* **Loading from MappingOptionsCache**: a :code:`MappingOptionsCache` provides
  a simple interface to load the best options from a previous tuning run.

* **Autotuning**: A kernel can be autotuned for fixed input tensor sizes.
  Optionally the best performing options can be cached to a file and reused to
  compile and run a TC operation.


Loading from cache
------------------

To load the best options from a previously saved :code:`MappingOptionsCache`
object, one can reconstruct it explicitly from a filename and load the best
(top-1) options given a TC string, an entry_point and a tuple of input
tensors as such:

    .. code-block:: python

        # Setup code for mm and tensors as above
        cache = tc.MappingOptionsCache(cache_filename)
        best_options, = cache.load(mm, "matmul", (A, B), 1)
        matmul = tc.define(mm, "matmul")
        matmul.compile(A, B, mapping_options=best_options)
        C, = matmul(A, B)

Autotuning
----------

Tuning can be achieved by constructing a TC and calling :code:`tune` on it.
The MappingOptions from which tuning starts is determined from the optional
parameters as follows:
    1. if starting_options is specified, it takes precedence;
    2. otherwise if load_from_cache is True, the best options for the
       current TC and input sizes are fetched from the backing
       self.cache_filename;
    3. if none of steps 1. or 2. above yield a MappingOptions,
       tuning starts from :code:`tc.MappingOptions('naive')`

    .. code-block:: python

        # Setup code for mm and tensors as above
        matmul = tc.define(mm, "matmul", cache_filename="some_file_name")
        best_options = matmul.tune(A, B, load_from_cache=True)
        matmul.compile(A, B, mapping_options = best_options)

    .. note::

       A tuning run can be aborted by sending the SIGINT signal (Ctrl+C). In
       that case, the compilation and evaluation jobs currently in flight will
       be flushed, but no new compilation job will be created. Once the jobs in
       flight are flushed, saving to cache occurs (if requested) and the best
       :code:`tc.MappingOptions` found so far will be returned.

Tuning behavior can be modified by defining the TC with an optional
:code:`tuner_config` parameter constructed as such:
:code:`tuner_config=tc.TunerConfig(threads=5, generations=3, pop_size=5)`.
For the list of configurable parameters and their defaults, one can
query :code:`help(tc.TunerConfig)`.

    .. note::

       By providing a fixed filename and calling short tuning runs over
       multiple executions with load_from_cache=True and store_to_cache=True,
       one can effectively reinforce the tuning process over time without
       paying a longer startup cost.

Fixed TC, varying input sizes
-----------------------------

A TC definition can be reused but needs to be recompiled for different size
combinations.

.. code-block:: python

    # Setup code for mm and tensors as above
    matmul = tc.define(mm, name="matmul")
    A1, B1 = torch.randn(300, 400).cuda(), torch.randn(400, 500).cuda()
    matmul.compile(A1, B1, mapping_options=best_options)
    C1, = matmul(A1, B1)

    # different input sizes
    A2, B2 = torch.randn(320, 450).cuda(), torch.randn(450, 300).cuda()
    matmul.compile(A2, B2, mapping_options=best_options)
    C2, = matmul(A2, B2)

    .. note::

        While we recommend tuning independently for each TC and input size
        variation, the best options found for a particular TC and input size
        combination may transfer well to another input size (especially if
        sizes are close and the kernels exhibit the same type of bottlenecs;
        i.e. memory-bound, latency-bound, instruction-issue-bound,
        compute-bound).

Implicit compilation
--------------------

TODO


Multiple TC definitions
-----------------------

If one wants to define all of TCs in one string and later use that string
for running different operations, one can define a :code:`lang` variable that
holds the TC definition for all operations.
Each time one wants to run a different operation, one can make a new TC object
by calling :code:`tc.define` on the :code:`lang` variable, specify the
:code:`entry_point` corresponding to the operation definition and obtain the
kernel implementing that operation:

.. code-block:: python

    import tensor_comprehensions as tc
    import torch
    lang = """
    def matmul(float(M, K) A, float(K, N) B) -> (C) {
        C(m, n) +=! A(m, r_k) * B(r_k, n)
    }
    def abs(float(M, N) A) -> (O) {
        O(m, n) = fabs(A(m, n))
    }
    """
    matmul = tc.define(lang, "matmul")
    A, B = torch.randn(3, 4).cuda(), torch.randn(4, 5).cuda()
    matmul.compile(A, B, mapping_options='naive')
    C, = matmul(A, B)

    abs = tc.define(lang, "abs")
    A = torch.randn(3, 4).cuda()
    abs.compile(A, mapping_options='naive')
    O, = abs(A)

.. note::


Writing layers with scalars
---------------------------

The TC mapper requires statically affine tensor indexing functions.
Without getting into deeper details, the dependence analysis process is
significantly simplified and can be represented exactly.
As a consequence, tensor subscripts should avoid multiplications
between an unknown parametric quantity and an index variable.
In practice this may require writing different TC versions for different stride
and kernel sizes. A simple workaround woud be for TC to provide a templating
mechanism.
A simple way to achieve the same effect is to dynamically perform string
substitutions based on runtime values by formatting the TC string with python
regular expressions:

    .. code-block:: python

        import tensor_comprehensions as tc
        import torch
        import re
        tc_str="""
        def avgpool(float(B, C, H, W) input) -> (output) {
            output(b, c, h, w) +=! input(b, c, h * <sH> + r_kh, w * <sW> + r_kw) / (<kH> * <kW>)
                where r_kh in 0:<kH>, r_kw in 0:<kW>
        }
        """
        sH, sW, kH, kW = 1, 1, 2, 2
        tc_str = re.sub('<sh>', str(sH), tc_str)
        tc_str = re.sub('<sw>', str(sW), tc_str)
        tc_str = re.sub('<kH>', str(kH), tc_str)
        tc_str = re.sub('<kW>', str(kW), tc_str)
        avgpool = tc.define(tc_str, "avgpool")
        inp = torch.ones(1, 1, 4, 4).cuda()
        avgpool.compile(inp, mapping_options='naive')
        out, = avgpool(inp)

Built-in Functions
------------------

TC allows using CUDA built-in functions as well when defining the TC operations.
During execution, the CUDA API will be called for those built-in functions. For example,
asusme one wants to use :code:`fmaxf` CUDA function in TC:

    .. code-block:: python

        import tensor_comprehensions as tc
        import torch
        tc_str = """
        def relu(float(B,M) I) -> (O) {
            O(b, m) = fmaxf(I(b, m), 0)
        }
        """
        relu = tc.define(tc_str, entry_point="relu", fallback=tc.MappingOptions('naive'))
        inp = torch.randn(100, 128).cuda()
        relu.compile(inp, mapping_options='naive')
        O, = relu(inp)

TC only supports a subset of built-in CUDA functions.
Built-in functions supported in TC are listed `here <https://github.com/facebookresearch/TensorComprehensions/blob/master/tc/core/libraries.h#L67>`_.
Documentation
for these functions is available as part of the official CUDA documentation `here <http://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE>`_.
