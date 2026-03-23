"""Public API for independent IBR algorithm package.

Top-level imports expose the PyTorch backend. Use
``from ibr_defused_algo.tensorflow_models import ...`` for TensorFlow/Keras models.
"""

from .models import DefusedIBR5IBR6, FusedIBR5IBR6, IBR5Net, IBR6Net


def get_tensorflow_models():
	"""Return TensorFlow/Keras model classes with a clear dependency error."""

	try:
		from .tensorflow_models import DefusedIBR5IBR6 as TFDefusedIBR5IBR6
		from .tensorflow_models import FusedIBR5IBR6 as TFFusedIBR5IBR6
		from .tensorflow_models import IBR5Net as TFIBR5Net
		from .tensorflow_models import IBR6Net as TFIBR6Net
	except Exception as exc:  # pragma: no cover - depends on optional tensorflow install
		raise ImportError(
			"TensorFlow backend is unavailable. Install with: "
			"pip install 'ibr-defused-algorithms[tensorflow]'"
		) from exc

	return TFIBR5Net, TFIBR6Net, TFDefusedIBR5IBR6, TFFusedIBR5IBR6


__all__ = [
	"IBR5Net",
	"IBR6Net",
	"DefusedIBR5IBR6",
	"FusedIBR5IBR6",
	"get_tensorflow_models",
]
