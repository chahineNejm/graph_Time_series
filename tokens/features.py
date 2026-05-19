"""Feature extraction, encoder, and decoder tokens."""

import numpy as np
from ..token import Token, _shapes


class FeatRaw(Token):
    name = "feat_raw"
    token_class = "feature"
    reads = ["cleaned"]
    writes = ["model_input"]
    description = "Use cleaned histories as model input"

    def apply(self, state):
        state.features["model_input"] = state.features["cleaned"].copy()
        state.log_step(self.name,
                       _shapes(state, ["cleaned"]),
                       {"model_input": state.features["model_input"].shape})
        state.token_sequence.append(self.name)
        return state


class FeatFFTEncode(Token):
    name = "fft_encode"
    token_class = "encoder"
    reads = ["cleaned"]
    writes = ["model_input", "fft_phase"]
    description = "FFT encode: magnitude as features, store phase for decoder"

    def apply(self, state):
        X = state.features["cleaned"]
        F = np.fft.rfft(X, axis=1)
        state.features["model_input"] = np.abs(F).astype(np.float32)
        state.features["fft_phase"] = np.angle(F).astype(np.float32)
        state.log_step(self.name,
                       _shapes(state, ["cleaned"]),
                       {k: state.features[k].shape for k in self.writes})
        state.token_sequence.append(self.name)
        return state


class FeatFFTDecode(Token):
    name = "fft_decode"
    token_class = "decoder"
    reads = ["fft_phase"]
    writes = []
    description = "STUB — inverse FFT reconstruction from predicted magnitudes + stored phase"

    def apply(self, state):
        # STUB: implement inverse transform here
        # To fill in:
        #   mag = state.last_prediction()        # predicted magnitudes
        #   phase = state.features["fft_phase"]
        #   F = mag * np.exp(1j * phase)
        #   decoded = np.fft.irfft(F, axis=1)
        #   state.prediction_stack[-1] = decoded
        #   state.update_residual()  (add this method if needed)
        state.log_step(self.name,
                       _shapes(state, ["fft_phase"]),
                       {})
        state.token_sequence.append(self.name)
        return state
