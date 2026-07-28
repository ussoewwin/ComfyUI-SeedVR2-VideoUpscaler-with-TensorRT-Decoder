import os
import locale
import inspect
import textwrap
from functools import partial
from pathlib import Path


def _safe_windows_decode_args() -> tuple:
    """Encoding + errors=replace for MSVC/OEM bytes on Japanese Windows (cp932)."""
    try:
        enc = locale.getpreferredencoding(False) or "utf-8"
    except Exception:
        enc = "utf-8"
    return (enc, "replace")


def _load_template_utf8(name: str, template_dir: Path) -> str:
    """UTF-8 open for inductor jinja templates (locale cp932 breaks on UTF-8 bytes)."""
    with open(template_dir / f"{name}.py.jinja", encoding="utf-8") as f:
        return f.read()


def _patch_inductor_load_template_utf8() -> None:
    """
    Root cause of:
      UnicodeDecodeError: 'cp932' codec can't decode byte 0x94 in position 618

    torch._inductor.utils.load_template uses open() without encoding= → locale
    cp932 on Japanese Windows. Jinja templates under torch/_inductor are UTF-8
    (e.g. cutedsl_mm_grouped.py.jinja). Failure happens at inductor import time
    when mm_grouped.py calls load_kernel_template(...).

    Must patch utils.load_template BEFORE torch._inductor.kernel.mm_common
    creates functools.partial(load_template, ...). If mm_common is already
    imported, rebind its partials to the UTF-8 loader.
    """
    try:
        import torch._inductor.utils as inductor_utils

        inductor_utils.load_template = _load_template_utf8  # type: ignore[assignment]
    except Exception as e:
        print(f"[SeedVR2] Warning: Could not patch inductor load_template: {e}")
        return

    # Rebind partials if kernel helpers already imported with the old function
    try:
        import sys

        mm_common = sys.modules.get("torch._inductor.kernel.mm_common")
        if mm_common is not None:
            if hasattr(mm_common, "_KERNEL_TEMPLATE_DIR"):
                mm_common.load_kernel_template = partial(
                    _load_template_utf8,
                    template_dir=mm_common._KERNEL_TEMPLATE_DIR,
                )
            if hasattr(mm_common, "_KERNEL_TEMPLATE_FB_DIR"):
                mm_common.load_fb_kernel_template = partial(
                    _load_template_utf8,
                    template_dir=mm_common._KERNEL_TEMPLATE_FB_DIR,
                )
    except Exception as e:
        print(f"[SeedVR2] Warning: Could not rebind mm_common load_kernel_template: {e}")

    try:
        import sys

        flex_common = sys.modules.get("torch._inductor.kernel.flex.common")
        if flex_common is not None and hasattr(flex_common, "_FLEX_TEMPLATE_DIR"):
            flex_common.load_flex_template = partial(
                _load_template_utf8,
                template_dir=flex_common._FLEX_TEMPLATE_DIR,
            )
    except Exception as e:
        print(f"[SeedVR2] Warning: Could not rebind flex load_flex_template: {e}")


def _patch_inductor_bmm_make_fallback_override() -> None:
    """
    Fix a recurring inductor assertion during VAE torch.compile:

      AssertionError: both a fallback and a decomp for same op: aten.<op>.default

    Known ops that trigger this with the installed torch:
      - aten.bmm.default   (registered decomp only handles outer-product [B,M,1]x[B,1,N])
      - aten.addmm.default  (registered decomp only handles specific shape cases)
      - aten.mm / aten.mv / aten.linear (same pattern: partial decomposition)

    torch._inductor.decomposition registers decompositions that handle only
    specific cases and return NotImplemented otherwise. In the general case
    inductor falls through to make_fallback(), but make_fallback() asserts
    `op not in check_decomps unless override_decomp=True`. With override_decomp
    left at its default (False) this assertion fires.

    Wrap make_fallback so that whenever the op appears in the active decomp
    table (either the explicitly passed get_decomp_fn or the global
    torch._inductor.lowering.decompositions) we transparently set
    override_decomp=True. The registered decomp is still tried first by
    make_fallback and returns NotImplemented for unsupported shapes, so this
    changes nothing semantically — it just silences the assertion that was
    guarding against accidental double-registration.

    IMPORTANT: torch._inductor.graph does `from torch._inductor.lowering
    import make_fallback` at module top, which binds the symbol into graph.py
    itself. Rebinding `lowering.make_fallback` is NOT enough if graph.py is
    already imported — graph.py keeps using the old reference. We must also
    rebind `graph.make_fallback` when present in sys.modules.
    """
    try:
        import torch._inductor.lowering as inductor_lowering
    except Exception as e:
        print(f"[SeedVR2] Warning: could not import inductor.lowering for make_fallback patch: {e}")
        return

    if getattr(inductor_lowering, "_seedvr2_bmm_override_patched", False):
        return

    _orig_make_fallback = inductor_lowering.make_fallback

    def _patched_make_fallback(op, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            # Compute the active decomposition table the same way make_fallback does:
            #   check_decomps = get_decomp_fn() if get_decomp_fn is not None else decompositions
            get_decomp_fn = kwargs.get("get_decomp_fn")
            if get_decomp_fn is not None:
                check_decomps = get_decomp_fn()
            else:
                check_decomps = inductor_lowering.decompositions

            # If the op is registered there (even as a partial decomp returning
            # NotImplemented), allow the fallback path by setting override_decomp.
            if op in check_decomps:
                kwargs["override_decomp"] = True
        except Exception:
            pass
        return _orig_make_fallback(op, *args, **kwargs)

    inductor_lowering.make_fallback = _patched_make_fallback  # type: ignore[assignment]
    inductor_lowering._seedvr2_bmm_override_patched = True  # type: ignore[attr-defined]

    # Rebind in graph.py too — its `from .lowering import make_fallback` already
    # captured the original reference if it was imported before us.
    try:
        import sys

        graph_mod = sys.modules.get("torch._inductor.graph")
        if graph_mod is not None and getattr(graph_mod, "make_fallback", None) is _orig_make_fallback:
            graph_mod.make_fallback = _patched_make_fallback  # type: ignore[assignment]
    except Exception as e:
        print(f"[SeedVR2] Warning: could not rebind graph.make_fallback: {e}")


def _fix_inductor_windows_encoding() -> None:
    """
    Harden torch inductor / cpp_extension for Japanese Windows (cp932).

    Two independent failure modes:

    A) open() of UTF-8 jinja templates with locale encoding (position ~618)
       → patch load_template to encoding="utf-8"

    B) MSVC / OEM subprocess stdout decoded strictly as oem/cp932
       → SUBPROCESS_DECODE_ARGS with errors="replace"

    SeedVR2 previously only patched (B). VAE torch.compile still failed on (A).

    Also fixes a separate VAE torch.compile failure:
      AssertionError: both a fallback and a decomp for same op: aten.bmm.default
    """
    if os.name != "nt":
        # bmm override is needed regardless of OS — apply on Linux too.
        _patch_inductor_bmm_make_fallback_override()
        return

    # Prefer English MSVC diagnostics when VS respects VSLANG
    os.environ.setdefault("VSLANG", "1033")

    # --- (0) jinja template UTF-8 open (actual VAE compile crash site) ---
    _patch_inductor_load_template_utf8()

    # --- (0b) bmm make_fallback override (VAE compile assertion) ---
    _patch_inductor_bmm_make_fallback_override()

    # --- (1) inductor cpp_builder ---
    try:
        import torch._inductor.cpp_builder as cpp_builder

        cpp_builder.SUBPROCESS_DECODE_ARGS = _safe_windows_decode_args()
    except Exception as e:
        print(f"[SeedVR2] Warning: Could not patch inductor SUBPROCESS_DECODE_ARGS: {e}")

    # --- (2) torch.utils.cpp_extension (common cp932 crash site) ---
    try:
        import torch.utils.cpp_extension as cpp_extension

        # Keep OEM code page (MSVC console) but never strict-fail on bad bytes
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("oem", "replace")
    except Exception as e:
        print(f"[SeedVR2] Warning: Could not patch cpp_extension SUBPROCESS_DECODE_ARGS: {e}")

    # --- (3) CppCompileError constructor: utf-8 without replace ---
    try:
        import torch._inductor.exc as inductor_exc

        _orig_init = inductor_exc.CppCompileError.__init__

        def _patched_cpp_compile_error_init(self, cmd, output):  # type: ignore[no-untyped-def]
            if isinstance(output, (bytes, bytearray)):
                output = bytes(output).decode("utf-8", errors="replace")
            _orig_init(self, cmd, output)

        inductor_exc.CppCompileError.__init__ = _patched_cpp_compile_error_init  # type: ignore[method-assign]
    except Exception as e:
        print(f"[SeedVR2] Warning: Could not patch CppCompileError: {e}")

    # --- (4) Legacy source rewrite for older inductor _run_compile_cmd ---
    try:
        import torch._inductor.cpp_builder as cpp_builder

        target_func = getattr(cpp_builder, "_run_compile_cmd", None)
        if target_func is None:
            return

        try:
            source = inspect.getsource(target_func)
        except OSError:
            return

        source = textwrap.dedent(source)
        replacements = (
            ('e.stdout.decode("utf-8")', 'e.stdout.decode("utf-8", errors="replace")'),
            ("e.stdout.decode('utf-8')", "e.stdout.decode('utf-8', errors='replace')"),
            ('e.output.decode("utf-8")', 'e.output.decode("utf-8", errors="replace")'),
        )
        new_source = source
        changed = False
        for old_code, new_code in replacements:
            if old_code in new_source and new_code not in new_source:
                new_source = new_source.replace(old_code, new_code)
                changed = True

        if not changed:
            return

        local_scope: dict = {}
        exec(new_source, cpp_builder.__dict__, local_scope)
        if "_run_compile_cmd" in local_scope:
            cpp_builder._run_compile_cmd = local_scope["_run_compile_cmd"]

    except Exception as e:
        print(f"[SeedVR2] Warning: Could not patch torch.inductor _run_compile_cmd: {e}")
