"""Validated input contract for Hugging Face Audio Spectrogram Transformer.

ASTFeatureExtractor produces 'input_values' (not 'input_features').
ASTForAudioClassification.forward receives 'input_values' as a named argument.
Keep training and inference on the same contract.
"""

def ast_input_values(extractor, waveform):
    """Convert 16-kHz mono samples to a single AST spectrogram tensor."""
    features = extractor(waveform, sampling_rate=16_000, return_tensors="pt")
    try:
        values = features["input_values"]
    except (KeyError, TypeError) as exc:
        raise ValueError("AST feature extractor did not return input_values.") from exc
    if (getattr(values, "ndim", None) != 3 or values.shape[0] != 1):
        raise ValueError("AST feature extractor must return one 3-D spectrogram batch.")
    return values


def forward_ast(model, input_values, *, labels=None):
    """Invoke the transformer using the documented input_values parameter."""
    if labels is None:
        return model(input_values=input_values)
    return model(input_values=input_values, labels=labels)
