"""llama.cpp + llama-swap management helpers."""


def main() -> int:
	# Import lazily to avoid preloading cli during `python -m llamacpp_stack.cli`.
	from .cli import main as cli_main

	return cli_main()


__all__ = ["main"]
